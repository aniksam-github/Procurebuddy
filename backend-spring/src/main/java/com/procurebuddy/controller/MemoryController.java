package com.procurebuddy.controller;

import com.procurebuddy.dto.request.MemoryRequest;
import com.procurebuddy.service.MemoryService;
import jakarta.validation.Valid;
import java.security.Principal;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class MemoryController {

    private final MemoryService memoryService;

    @PostMapping({"/api/memory", "/memory"})
    public Map<String, Object> saveMemory(@Valid @RequestBody MemoryRequest request, Principal principal) {
        return memoryService.upsertMemory(principal.getName(), null, request.getKey(), request.getValue());
    }

    @GetMapping({"/api/memory", "/memory"})
    public Map<String, Object> getMemory(@RequestParam(value = "key", required = false) String key, Principal principal) {
        return memoryService.getMemory(principal.getName(), null, key);
    }
}
