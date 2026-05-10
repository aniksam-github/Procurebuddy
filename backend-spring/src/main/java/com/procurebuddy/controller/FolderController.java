package com.procurebuddy.controller;

import com.procurebuddy.dto.request.CreateFolderRequest;
import com.procurebuddy.dto.request.MoveChatRequest;
import com.procurebuddy.service.ChatService;
import com.procurebuddy.service.FolderService;
import java.security.Principal;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/folders")
@RequiredArgsConstructor
public class FolderController {

    private final FolderService folderService;
    private final ChatService chatService;

    @PostMapping
    public Map<String, Object> createFolder(@RequestBody CreateFolderRequest request, Principal principal) {
        return folderService.createFolder(principal.getName(), null, request.getName());
    }

    @GetMapping
    public Map<String, Object> listFolders(Principal principal) {
        return folderService.listFolders(principal.getName(), null);
    }

    @PostMapping("/move")
    public Map<String, Object> moveChat(@RequestBody MoveChatRequest request, Principal principal) {
        return chatService.moveChatToFolder(request.getChatId(), request.getFolderId(), principal.getName(), null);
    }

    @DeleteMapping("/{folderId}")
    public Map<String, Object> deleteFolder(@PathVariable String folderId, Principal principal) {
        return folderService.deleteFolder(folderId, principal.getName(), null);
    }
}
