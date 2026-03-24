package com.procurebuddy.controller;

import com.procurebuddy.service.AdminService;
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
    public Map<String, Object> listDocuments(@RequestParam String email) {
        return adminService.listDocuments(email);
    }

    @GetMapping("/status")
    public Map<String, Object> status(@RequestParam String email) {
        return adminService.status(email);
    }

    @PostMapping("/upload")
    public Map<String, Object> upload(@RequestParam String email, @RequestParam("files") MultipartFile[] files) {
        return adminService.uploadDocuments(email, files);
    }

    @PostMapping("/reindex")
    public Map<String, Object> reindex(@RequestParam String email) {
        return adminService.reindexDocuments(email);
    }
}
