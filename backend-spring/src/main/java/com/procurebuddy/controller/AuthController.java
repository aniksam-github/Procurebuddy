package com.procurebuddy.controller;

import com.procurebuddy.dto.request.ChangePasswordRequest;
import com.procurebuddy.dto.request.LoginRequest;
import com.procurebuddy.dto.request.RegisterStartRequest;
import com.procurebuddy.dto.request.RegisterVerifyRequest;
import com.procurebuddy.dto.request.ResetPasswordRequest;
import com.procurebuddy.dto.request.TotpDisableRequest;
import com.procurebuddy.dto.request.TotpEnableRequest;
import com.procurebuddy.dto.request.TotpSetupRequest;
import com.procurebuddy.dto.request.TotpVerifyRequest;
import com.procurebuddy.dto.request.UpdateProfileRequest;
import com.procurebuddy.service.AuthService;
import jakarta.validation.Valid;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/register/start")
    public Map<String, Object> registerStart(@Valid @RequestBody RegisterStartRequest request) {
        return authService.registerStart(request.getEmail());
    }

    @PostMapping("/register/verify")
    public Map<String, Object> registerVerify(@Valid @RequestBody RegisterVerifyRequest request) {
        return authService.registerVerify(request.getEmail(), request.getOtp(), request.getPassword());
    }

    @PostMapping("/login")
    public AuthService.LoginResponse login(@Valid @RequestBody LoginRequest request) {
        return authService.login(request.getEmail(), request.getPassword());
    }

    @GetMapping("/status")
    public Map<String, Object> status(@RequestParam String email) {
        return authService.authStatus(email);
    }

    @PostMapping("/change-password")
    public Map<String, Object> changePassword(@Valid @RequestBody ChangePasswordRequest request) {
        return authService.changePassword(request.getEmail(), request.getNewPassword());
    }

    @PostMapping("/profile")
    public Map<String, Object> updateProfile(@Valid @RequestBody UpdateProfileRequest request) {
        return authService.updateProfile(
                request.getEmail(),
                request.getDisplayName(),
                request.getUsername(),
                request.getAvatarBase64()
        );
    }

    @PostMapping("/reset-password")
    public Map<String, Object> resetPassword(@Valid @RequestBody ResetPasswordRequest request) {
        return authService.resetPassword(request.getEmail());
    }

    @PostMapping("/totp/setup")
    public Map<String, Object> setupTotp(@Valid @RequestBody TotpSetupRequest request) {
        return authService.setupTotp(request.getEmail());
    }

    @PostMapping("/totp/enable")
    public Map<String, Object> enableTotp(@Valid @RequestBody TotpEnableRequest request) {
        return authService.enableTotp(request.getEmail(), request.getSecret(), request.getCode());
    }

    @PostMapping("/totp/verify")
    public Map<String, Object> verifyTotp(@Valid @RequestBody TotpVerifyRequest request) {
        return authService.verifyTotp(request.getEmail(), request.getCode());
    }

    @PostMapping("/totp/disable")
    public Map<String, Object> disableTotp(@Valid @RequestBody TotpDisableRequest request) {
        return authService.disableTotp(request.getEmail());
    }
}
