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
import java.security.Principal;
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
    public Map<String, Object> status(Principal principal) {
        return authService.authStatus(principal.getName());
    }

    @PostMapping("/change-password")
    public Map<String, Object> changePassword(@Valid @RequestBody ChangePasswordRequest request, Principal principal) {
        return authService.changePassword(principal == null ? null : principal.getName(), request.getNewPassword(), request.getLoginToken());
    }

    @PostMapping("/profile")
    public Map<String, Object> updateProfile(@Valid @RequestBody UpdateProfileRequest request, Principal principal) {
        return authService.updateProfile(
                principal.getName(),
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
    public Map<String, Object> setupTotp(@Valid @RequestBody TotpSetupRequest request, Principal principal) {
        return authService.setupTotp(principal.getName());
    }

    @PostMapping("/totp/enable")
    public Map<String, Object> enableTotp(@Valid @RequestBody TotpEnableRequest request, Principal principal) {
        return authService.enableTotp(principal.getName(), request.getSecret(), request.getCode());
    }

    @PostMapping("/totp/verify")
    public AuthService.LoginResponse verifyTotp(@Valid @RequestBody TotpVerifyRequest request) {
        return authService.verifyTotp(request.getEmail(), request.getCode(), request.getLoginToken());
    }

    @PostMapping("/totp/disable")
    public Map<String, Object> disableTotp(@Valid @RequestBody TotpDisableRequest request, Principal principal) {
        return authService.disableTotp(principal.getName());
    }
}
