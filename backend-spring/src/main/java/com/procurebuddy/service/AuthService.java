package com.procurebuddy.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.procurebuddy.security.JwtService;
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
import java.util.LinkedHashMap;
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
    private final JwtService jwtService;
    private final GoogleAuthenticator googleAuthenticator = new GoogleAuthenticator();
    private final SecureRandom secureRandom = new SecureRandom();

    private static final String CHALLENGE_PURPOSE_TOTP = "totp";
    private static final String CHALLENGE_PURPOSE_PASSWORD_CHANGE = "password_change";

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
        user.setDisplayName(defaultDisplayName(normalizedEmail));
        user.setUsername(defaultUsername(normalizedEmail));
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

        boolean isAdmin = isAdminEmail(user.getEmail());
        if (user.isMustChange()) {
            return LoginResponse.passwordChangeRequired(
                    user,
                    isAdmin,
                    jwtService.issueChallengeToken(user.getEmail(), CHALLENGE_PURPOSE_PASSWORD_CHANGE)
            );
        }
        if (user.isTotpEnabled()) {
            return LoginResponse.totpRequired(
                    user,
                    isAdmin,
                    jwtService.issueChallengeToken(user.getEmail(), CHALLENGE_PURPOSE_TOTP)
            );
        }

        return LoginResponse.authenticated(user, isAdmin, jwtService.issueAccessToken(user.getEmail(), isAdmin));
    }

    @Transactional(readOnly = true)
    public Map<String, Object> authStatus(String currentEmail) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(currentEmail))
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "User not found."));

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("success", true);
        response.put("email", user.getEmail());
        response.put("display_name", nullToEmpty(user.getDisplayName()));
        response.put("username", nullToEmpty(user.getUsername()));
        response.put("avatar_base64", nullToEmpty(user.getAvatarBase64()));
        response.put("must_change", user.isMustChange());
        response.put("totp_enabled", user.isTotpEnabled());
        response.put("is_admin", isAdminEmail(user.getEmail()));
        response.put("created_at", user.getCreatedAt() == null ? null : user.getCreatedAt().toString());
        return response;
    }

    @Transactional
    public Map<String, Object> changePassword(String currentEmail, String newPassword, String loginToken) {
        if (!PasswordRules.isStrong(newPassword)) {
            throw new ApiException(HttpStatus.BAD_REQUEST, PasswordRules.REQUIREMENT_MESSAGE);
        }

        UserEntity user = resolveUserForPasswordChange(currentEmail, loginToken);
        user.setPasswordHash(passwordEncoder.encode(newPassword));
        user.setMustChange(false);
        userRepository.save(user);
        return Map.of("success", true, "message", "Password changed successfully.");
    }

    @Transactional
    public Map<String, Object> updateProfile(String currentEmail, String displayName, String username, String avatarBase64) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(currentEmail))
                .orElseThrow(() -> new ApiException(HttpStatus.BAD_REQUEST, "User not found."));

        String sanitizedDisplayName = sanitizeDisplayName(displayName);
        String sanitizedUsername = sanitizeUsername(username);
        user.setDisplayName(sanitizedDisplayName);
        user.setUsername(sanitizedUsername);
        user.setAvatarBase64(sanitizeAvatar(avatarBase64));
        userRepository.save(user);

        return Map.of(
                "success", true,
                "message", "Profile updated successfully.",
                "profile", Map.of(
                        "email", user.getEmail(),
                        "display_name", user.getDisplayName(),
                        "username", user.getUsername(),
                        "avatar_base64", user.getAvatarBase64() == null ? "" : user.getAvatarBase64()
                )
        );
    }

    @Transactional
    public Map<String, Object> resetPassword(String email) {
        String normalizedEmail = UserResolver.normalizeEmail(email);
        UserEntity user = userRepository.findByEmailIgnoreCase(normalizedEmail).orElse(null);
        if (user == null) {
            return Map.of("success", true, "message", "If an account exists for that email, a temporary password has been sent.");
        }

        String tempPassword = generateTempPassword();
        user.setPasswordHash(passwordEncoder.encode(tempPassword));
        user.setMustChange(true);
        userRepository.save(user);
        otpMailService.sendTemporaryPassword(user.getEmail(), tempPassword);
        return Map.of("success", true, "message", "If an account exists for that email, a temporary password has been sent.");
    }

    @Transactional
    public Map<String, Object> setupTotp(String currentEmail) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(currentEmail))
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
    public Map<String, Object> enableTotp(String currentEmail, String secret, String code) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(currentEmail))
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
    public LoginResponse verifyTotp(String email, String code, String loginToken) {
        String normalizedEmail = UserResolver.normalizeEmail(email);
        if (loginToken == null || loginToken.isBlank()) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Login challenge expired. Please sign in again.");
        }

        String tokenEmail;
        try {
            tokenEmail = UserResolver.normalizeEmail(jwtService.extractChallengeSubject(loginToken, CHALLENGE_PURPOSE_TOTP));
        } catch (Exception ex) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Login challenge expired. Please sign in again.");
        }
        if (!tokenEmail.equals(normalizedEmail)) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Login challenge does not match this account.");
        }

        UserEntity user = userRepository.findByEmailIgnoreCase(normalizedEmail).orElse(null);
        boolean valid = user != null && user.isTotpEnabled() && user.getTotpSecret() != null && verifyTotpCode(user.getTotpSecret(), code);
        if (!valid) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Invalid or expired TOTP code.");
        }
        boolean isAdmin = isAdminEmail(user.getEmail());
        return LoginResponse.authenticated(user, isAdmin, jwtService.issueAccessToken(user.getEmail(), isAdmin));
    }

    @Transactional
    public Map<String, Object> disableTotp(String currentEmail) {
        UserEntity user = userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(currentEmail))
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

    private UserEntity resolveUserForPasswordChange(String currentEmail, String loginToken) {
        if (currentEmail != null && !currentEmail.isBlank()) {
            return userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(currentEmail))
                    .orElseThrow(() -> new ApiException(HttpStatus.BAD_REQUEST, "User not found."));
        }
        if (loginToken == null || loginToken.isBlank()) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Authentication required.");
        }
        String tokenEmail;
        try {
            tokenEmail = jwtService.extractChallengeSubject(loginToken, CHALLENGE_PURPOSE_PASSWORD_CHANGE);
        } catch (Exception ex) {
            throw new ApiException(HttpStatus.UNAUTHORIZED, "Password reset session expired. Please sign in again.");
        }
        return userRepository.findByEmailIgnoreCase(UserResolver.normalizeEmail(tokenEmail))
                .orElseThrow(() -> new ApiException(HttpStatus.BAD_REQUEST, "User not found."));
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
            @JsonProperty("display_name") String displayName,
            String username,
            @JsonProperty("avatar_base64") String avatarBase64,
            @JsonProperty("must_change") boolean mustChange,
            @JsonProperty("totp_required") boolean totpRequired,
            @JsonProperty("totp_enabled") boolean totpEnabled,
            @JsonProperty("is_admin") boolean isAdmin,
            String token,
            @JsonProperty("access_token") String accessToken,
            @JsonProperty("login_token") String loginToken
    ) {
        public static LoginResponse authenticated(UserEntity user, boolean isAdmin, String token) {
            return new LoginResponse(
                    true,
                    user.getEmail(),
                    nullToEmpty(user.getDisplayName()),
                    nullToEmpty(user.getUsername()),
                    nullToEmpty(user.getAvatarBase64()),
                    user.isMustChange(),
                    false,
                    user.isTotpEnabled(),
                    isAdmin,
                    token,
                    token,
                    ""
            );
        }

        public static LoginResponse totpRequired(UserEntity user, boolean isAdmin, String loginToken) {
            return new LoginResponse(
                    true,
                    user.getEmail(),
                    nullToEmpty(user.getDisplayName()),
                    nullToEmpty(user.getUsername()),
                    nullToEmpty(user.getAvatarBase64()),
                    false,
                    true,
                    true,
                    isAdmin,
                    "",
                    "",
                    loginToken
            );
        }

        public static LoginResponse passwordChangeRequired(UserEntity user, boolean isAdmin, String loginToken) {
            return new LoginResponse(
                    true,
                    user.getEmail(),
                    nullToEmpty(user.getDisplayName()),
                    nullToEmpty(user.getUsername()),
                    nullToEmpty(user.getAvatarBase64()),
                    true,
                    false,
                    user.isTotpEnabled(),
                    isAdmin,
                    "",
                    "",
                    loginToken
            );
        }
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }

    private String defaultDisplayName(String email) {
        String local = email == null ? "ProcureBuddy User" : email.split("@")[0];
        String[] parts = local.replaceAll("[^A-Za-z0-9._-]", " ").split("[._-]+");
        StringBuilder builder = new StringBuilder();
        for (String part : parts) {
            if (part == null || part.isBlank()) continue;
            if (!builder.isEmpty()) {
                builder.append(' ');
            }
            builder.append(Character.toUpperCase(part.charAt(0)));
            if (part.length() > 1) {
                builder.append(part.substring(1));
            }
        }
        return builder.isEmpty() ? "ProcureBuddy User" : builder.toString();
    }

    private String defaultUsername(String email) {
        if (email == null || email.isBlank()) {
            return "user";
        }
        return sanitizeUsername(email.split("@")[0]);
    }

    private String sanitizeDisplayName(String value) {
        if (value == null) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Display name is required.");
        }
        String normalized = value.strip().replaceAll("\\s+", " ");
        if (normalized.length() < 2) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Display name must be at least 2 characters.");
        }
        if (normalized.length() > 80) {
            normalized = normalized.substring(0, 80).trim();
        }
        return normalized;
    }

    private String sanitizeUsername(String value) {
        if (value == null) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Username is required.");
        }
        String normalized = value.strip().toLowerCase().replaceAll("[^a-z0-9._-]", "");
        if (normalized.length() < 3) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Username must be at least 3 characters.");
        }
        if (normalized.length() > 32) {
            normalized = normalized.substring(0, 32);
        }
        return normalized;
    }

    private String sanitizeAvatar(String avatarBase64) {
        if (avatarBase64 == null || avatarBase64.isBlank()) {
            return null;
        }
        String normalized = avatarBase64.strip();
        if (normalized.length() > 2_000_000) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Profile photo is too large.");
        }
        if (!normalized.startsWith("data:image/")) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Profile photo must be a valid image.");
        }
        return normalized;
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
