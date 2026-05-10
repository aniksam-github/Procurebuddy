package com.procurebuddy.service;

import com.procurebuddy.exception.ApiException;
import jakarta.mail.internet.MimeMessage;
import java.nio.charset.StandardCharsets;
import lombok.RequiredArgsConstructor;
import org.springframework.core.env.Environment;
import org.springframework.http.HttpStatus;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class OtpMailService {

    private final JavaMailSender mailSender;
    private final Environment environment;

    public void sendRegistrationOtp(String email, String otp) {
        String username = requireMailConfiguration();
        sendPlainText(email, username, "ProcureBuddy registration OTP",
                "Your ProcureBuddy registration OTP is:\n\n" + otp + "\n\n"
                        + "It expires in 10 minutes. Do not share it with anyone.");
    }

    public void sendTemporaryPassword(String email, String tempPassword) {
        String username = requireMailConfiguration();
        sendPlainText(email, username, "ProcureBuddy temporary password",
                "A temporary password was requested for your ProcureBuddy account.\n\n"
                        + "Temporary password: " + tempPassword + "\n\n"
                        + "Use it to sign in and change your password immediately.");
    }

    private String requireMailConfiguration() {
        String host = environment.getProperty("spring.mail.host", "");
        String username = environment.getProperty("spring.mail.username", "");
        String password = environment.getProperty("spring.mail.password", "");

        if (host.isBlank() || username.isBlank() || password.isBlank()) {
            throw new ApiException(
                    HttpStatus.INTERNAL_SERVER_ERROR,
                    "Email is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, and SMTP_PASS before using email-based auth flows."
            );
        }
        return username;
    }

    private void sendPlainText(String email, String from, String subject, String body) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, false, StandardCharsets.UTF_8.name());
            helper.setTo(email);
            helper.setFrom(from);
            helper.setSubject(subject);
            helper.setText(body, false);
            mailSender.send(message);
        } catch (Exception ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to send email to " + email + ".");
        }
    }
}
