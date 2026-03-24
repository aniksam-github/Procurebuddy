package com.procurebuddy.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.zxing.BarcodeFormat;
import com.google.zxing.client.j2se.MatrixToImageWriter;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.qrcode.QRCodeWriter;
import com.procurebuddy.config.ProcureBuddyProperties;
import com.procurebuddy.entity.PendingOtpEntity;
import com.procurebuddy.entity.UserEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.PendingOtpRepository;
import com.procurebuddy.repository.UserRepository;
import com.procurebuddy.util.PasswordRules;
import com.procurebuddy.util.UserResolver;
import com.warrenstrange.googleauth.GoogleAuthenticator;
import java.io.ByteArrayOutputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.time.LocalDateTime;
import java.util.Base64;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class AuthService {

    private static final int OTP_EXPIRY_SECS = 600;

    private final UserRepository userRepository;
    private final PendingOtpRepository pendingOtpRepository;
    private final ProcureBuddyProperties properties;
    private final OtpMailService otpMailService;
    private final BCryptPasswordEncoder passwordEncoder;
    private final GoogleAuthenticator googleAuthenticator = new GoogleAuthenticator();
    private final SecureRandom secureRandom = new SecureRandom();

    @Transactional
    public Map<String, Object> registerStart(String email) {
        String normalizedEmail = UserResolver.normalizeEmail(email);
        if (!isOfficialEmail(normalizedEmail)) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Please use a CBRI, Outlook, or Gmail email address.");
        }
        if (userRepository.findByEmailIgnoreCase(normalizedEmail).isPresent()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "An account with this email already exists.");
        }

        PendingOtpEntity pendingOtp = pendingOtpRepository.findByEmailIgnoreCase(normalizedEmail)
                .orElseGet(PendingOtpEntity::new);
        pendingOtp.setEmail(normalizedEmail);
        pendingOtp.setOtp(generateOtp());
        pendingOtp.setExpiresAt(LocalDateTime.now().plusSeconds(OTP_EXPIRY_SECS));
        pendingOtpRepository.save(pendingOtp);

        otpMailService.sendRegistrationOtp(normalizedEmail, pendingOtp.getOtp());
        return Map.of("success", true, "message", "OTP sent to your email.");
    }

    @Transactional
    public Map<String, Object> registerVerify(String email, String otp, String password) {
        String normalizedEmail = UserResolver.normalizeEmail(email);
        PendingOtpEntity pendingOtp = pendingOtpRepository.findByEmailIgnoreCase(normalizedEmail)
                .orElseThrow(() -> new ApiException(HttpStatus.BAD_REQUEST, "No pending registration for this email."));

        if (LocalDateTime.now().isAfter(pendingOtp.getExpiresAt())) {
            pendingOtpRepository.delete(pendingOtp);
            throw new ApiException(HttpStatus.BAD_REQUEST, "OTP has expired. Please request a new one.");
        }
        if (!pendingOtp.getOtp().equals(otp)) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Invalid OTP.");
        }
        if (!PasswordRules.isStrong(password)) {
            throw new ApiException(HttpStatus.BAD_REQUEST, PasswordRules.REQUIREMENT_MESSAGE);
        }

        UserEntity user = new UserEntity();
        user.setEmail(normalizedEmail);
        user.setPasswordHash(passwordEncoder.encode(password));
        user.setMustChange(false);
        user.setTotpEnabled(false);
        userRepository.save(user);
        pendingOtpRepository.delete(pendingOtp);

        return Map.of("success", true, "message", "Account created successfully.");
    }

    @Transactional(readOnly = true)
    public LoginResponse login(String email, String password) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElseThrow(() -> new ApiException(HttpStatus.UNAUTHORIZED, "Invalid email or password."));

        if (!passwordEncoder.matches(password, user.getPasswordHash())) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Invalid email or password.");
        }

        return LoginResponse.from(user, isAdminEmail(user.getEmail()));
    }

    @Transactional(readOnly = true)
    public Map<String, Object> authStatus(String email) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "User not found."));

        return Map.of(
                "success", true,
                "email", user.getEmail(),
                "must_change", user.isMustChange(),
                "totp_enabled", user.isTotpEnabled(),
                "is_admin", isAdminEmail(user.getEmail()),
                "created_at", user.getCreatedAt()
        );
    }

    @Transactional
    public Map<String, Object> changePassword(String email, String newPassword) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElseThrow(() -> new ApiException(HttpStatus.BAD_REQUEST, "User not found."));

        if (!PasswordRules.isStrong(newPassword)) {
            throw new ApiException(HttpStatus.BAD_REQUEST, PasswordRules.REQUIREMENT_MESSAGE);
        }

        user.setPasswordHash(passwordEncoder.encode(newPassword));
        user.setMustChange(false);
        userRepository.save(user);
        return Map.of("success", true, "message", "Password changed successfully.");
    }

    @Transactional
    public Map<String, Object> resetPassword(String email) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "User not found."));

        String tempPassword = generateTempPassword();
        user.setPasswordHash(passwordEncoder.encode(tempPassword));
        user.setMustChange(true);
        userRepository.save(user);
        return Map.of("success", true, "temp_password", tempPassword);
    }

    @Transactional
    public Map<String, Object> setupTotp(String email) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "User not found."));

        String secret = googleAuthenticator.createCredentials().getKey();
        user.setPendingTotpSecret(secret);
        userRepository.save(user);

        return Map.of(
                "success", true,
                "secret", secret,
                "qr_base64", buildQrCode(secret, user.getEmail())
        );
    }

    @Transactional
    public Map<String, Object> enableTotp(String email, String secret, String code) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElseThrow(() -> new ApiException(HttpStatus.BAD_REQUEST, "User not found."));

        String effectiveSecret = (secret != null && !secret.isBlank()) ? secret : user.getPendingTotpSecret();
        if (effectiveSecret == null || effectiveSecret.isBlank() || !verifyTotpCode(effectiveSecret, code)) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Invalid TOTP code. Please scan again.");
        }

        user.setTotpEnabled(true);
        user.setTotpSecret(effectiveSecret);
        user.setPendingTotpSecret(null);
        userRepository.save(user);

        return Map.of("success", true, "message", "Two-factor authentication enabled.");
    }

    @Transactional(readOnly = true)
    public Map<String, Object> verifyTotp(String email, String code) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElse(null);
        boolean valid = user != null
                && user.isTotpEnabled()
                && user.getTotpSecret() != null
                && verifyTotpCode(user.getTotpSecret(), code);
        if (!valid) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Invalid or expired TOTP code.");
        }
        return Map.of("success", true, "message", "TOTP verified.");
    }

    @Transactional
    public Map<String, Object> disableTotp(String email) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(email))
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "User not found."));

        user.setTotpEnabled(false);
        user.setTotpSecret(null);
        user.setPendingTotpSecret(null);
        userRepository.save(user);

        return Map.of("success", true, "message", "Two-factor authentication disabled.");
    }

    public boolean isAdminEmail(String email) {
        return UserResolver.normalizeEmail(email).equals(UserResolver.normalizeEmail(properties.getAdminEmail()));
    }

    private boolean isOfficialEmail(String email) {
        return email.endsWith(".cbri@csir.res.in")
                || email.endsWith("@outlook.com")
                || email.endsWith("@gmail.com");
    }

    private String generateOtp() {
        return String.format("%06d", secureRandom.nextInt(1_000_000));
    }

    private String generateTempPassword() {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(RandomString.randomBytes(10)).substring(0, 11);
    }

    private boolean verifyTotpCode(String secret, String code) {
        try {
            int numericCode = Integer.parseInt(code);
            return googleAuthenticator.authorize(secret, numericCode);
        } catch (NumberFormatException ex) {
            return false;
        }
    }

    private String buildQrCode(String secret, String email) {
        try {
            String issuer = "CBRI ProcureBuddy";
            String label = URLEncoder.encode(issuer + ":" + email, StandardCharsets.UTF_8);
            String encodedIssuer = URLEncoder.encode(issuer, StandardCharsets.UTF_8);
            String uri = "otpauth://totp/" + label + "?secret=" + secret + "&issuer=" + encodedIssuer;
            BitMatrix matrix = new QRCodeWriter().encode(uri, BarcodeFormat.QR_CODE, 240, 240);
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            MatrixToImageWriter.writeToStream(matrix, "PNG", output);
            return Base64.getEncoder().encodeToString(output.toByteArray());
        } catch (Exception ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to generate TOTP QR.");
        }
    }

    public record LoginResponse(
            boolean success,
            String email,
            @JsonProperty("must_change") boolean mustChange,
            @JsonProperty("totp_required") boolean totpRequired,
            @JsonProperty("totp_enabled") boolean totpEnabled,
            @JsonProperty("is_admin") boolean isAdmin
    ) {
        public static LoginResponse from(UserEntity user, boolean isAdmin) {
            return new LoginResponse(true, user.getEmail(), user.isMustChange(), user.isTotpEnabled(), user.isTotpEnabled(), isAdmin);
        }
    }

    private static final class RandomString {
        private RandomString() {
        }

        private static byte[] randomBytes(int length) {
            byte[] bytes = new byte[length];
            new SecureRandom().nextBytes(bytes);
            return bytes;
        }
    }
}
