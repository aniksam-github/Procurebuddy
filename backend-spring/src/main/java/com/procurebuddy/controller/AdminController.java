package com.procurebuddy.controller;

import com.procurebuddy.service.AdminService;
import java.security.Principal;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
public class AdminController {

    private final AdminService adminService;

    @GetMapping("/documents")
    public Map<String, Object> listDocuments(Principal principal) {
        return adminService.listDocuments(principal.getName());
    }

    @GetMapping("/status")
    public Map<String, Object> status(Principal principal) {
        return adminService.status(principal.getName());
    }

    @PostMapping("/upload")
    public Map<String, Object> upload(Principal principal, @RequestParam("files") MultipartFile[] files) {
        return adminService.uploadDocuments(principal.getName(), files);
    }

    @PostMapping("/reindex")
    public Map<String, Object> reindex(Principal principal) {
        return adminService.reindexDocuments(principal.getName());
    }
}
