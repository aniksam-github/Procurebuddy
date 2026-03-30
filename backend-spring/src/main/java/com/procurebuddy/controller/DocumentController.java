package com.procurebuddy.controller;

import com.procurebuddy.dto.request.DocumentSearchRequest;
import com.procurebuddy.service.DocumentService;
import jakarta.validation.Valid;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequiredArgsConstructor
public class DocumentController {

    private final DocumentService documentService;

    @PostMapping({"/api/documents/upload", "/documents/upload"})
    public Map<String, Object> uploadDocuments(
            @RequestParam(value = "file", required = false) MultipartFile file,
            @RequestParam(value = "files", required = false) MultipartFile[] files
    ) {
        return documentService.storeDocuments(file, files);
    }

    @PostMapping({"/api/documents/search", "/documents/search"})
    public Map<String, Object> searchDocuments(@Valid @RequestBody DocumentSearchRequest request) {
        return documentService.searchDocuments(request.getQuery());
    }
}
