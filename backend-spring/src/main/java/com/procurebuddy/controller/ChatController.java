package com.procurebuddy.controller;

import com.procurebuddy.dto.request.PinChatRequest;
import com.procurebuddy.dto.request.SendMessageRequest;
import com.procurebuddy.service.ChatExportService;
import com.procurebuddy.service.ChatService;
import jakarta.validation.Valid;
import java.security.Principal;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping
@RequiredArgsConstructor
public class ChatController {

    private final ChatService chatService;
    private final ChatExportService chatExportService;

    @GetMapping({"/api/chats", "/chats"})
    public Map<String, Object> listChats(
            @RequestParam(value = "page", required = false) Integer page,
            @RequestParam(value = "size", required = false) Integer size,
            Principal principal
    ) {
        return chatService.listChats(principal.getName(), page, size);
    }

    @GetMapping({"/api/chats/{chatId}", "/chats/{chatId}"})
    public Map<String, Object> getChat(
            @PathVariable String chatId,
            @RequestParam(value = "page", required = false) Integer page,
            @RequestParam(value = "size", required = false) Integer size,
            Principal principal
    ) {
        return chatService.getChat(chatId, principal.getName(), page, size);
    }

    @PostMapping({"/api/chats/{chatId}/message", "/chats/{chatId}/message"})
    public CompletableFuture<Map<String, Object>> sendMessage(
            @PathVariable String chatId,
            @Valid @RequestBody SendMessageRequest request,
            Principal principal
    ) {
        return chatService.sendMessageAsync(chatId, principal.getName(), request.getMessage());
    }

    @PostMapping({"/api/chats/{chatId}/regenerate", "/chats/{chatId}/regenerate"})
    public CompletableFuture<Map<String, Object>> regenerateResponse(
            @PathVariable String chatId,
            Principal principal
    ) {
        return chatService.regenerateLastResponseAsync(chatId, principal.getName());
    }

    @PostMapping("/chat/regenerate")
    public CompletableFuture<Map<String, Object>> regenerateResponseCompat(
            @RequestParam("chatId") String chatId,
            Principal principal
    ) {
        return chatService.regenerateLastResponseAsync(chatId, principal.getName());
    }

    @DeleteMapping({"/api/chats/{chatId}", "/chats/{chatId}"})
    public Map<String, Object> deleteChat(
            @PathVariable String chatId,
            Principal principal
    ) {
        return chatService.deleteChat(chatId, principal.getName(), null);
    }

    @PostMapping({"/api/chats/pin", "/chats/pin"})
    public Map<String, Object> pinChat(@RequestBody PinChatRequest request, Principal principal) {
        return chatService.pinChat(request.getChatId(), request.getPinned(), principal.getName(), null);
    }

    @GetMapping(value = "/api/chats/{chatId}/export", produces = MediaType.APPLICATION_PDF_VALUE)
    public ResponseEntity<byte[]> exportChat(
            @PathVariable String chatId,
            Principal principal
    ) {
        ChatExportService.ChatExportResult result = chatExportService.exportChatPdf(chatId, principal.getName());
        return ResponseEntity.ok()
                .header(
                        HttpHeaders.CONTENT_DISPOSITION,
                        ContentDisposition.attachment().filename(result.filename()).build().toString()
                )
                .contentType(MediaType.APPLICATION_PDF)
                .body(result.content());
    }
}
