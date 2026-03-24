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
        String host = environment.getProperty("spring.mail.host", "");
        String username = environment.getProperty("spring.mail.username", "");
        String password = environment.getProperty("spring.mail.password", "");
        String port = environment.getProperty("spring.mail.port", "587");

        if (host.isBlank() || username.isBlank() || password.isBlank()) {
            throw new ApiException(
                    HttpStatus.INTERNAL_SERVER_ERROR,
                    "OTP email is not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, and SMTP_PASS before registering."
            );
        }

        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, false, StandardCharsets.UTF_8.name());
            helper.setTo(email);
            helper.setFrom(username);
            helper.setSubject("ProcureBuddy registration OTP");
            helper.setText(
                    "Your ProcureBuddy registration OTP is:\n\n" + otp + "\n\n"
                            + "It expires in 10 minutes. Do not share it with anyone.",
                    false
            );
            mailSender.send(message);
        } catch (Exception ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to send OTP email to " + email + ".");
        }
    }
}
