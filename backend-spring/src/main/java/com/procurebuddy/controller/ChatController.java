package com.procurebuddy.controller;

import com.procurebuddy.dto.request.PinChatRequest;
import com.procurebuddy.dto.request.SendMessageRequest;
import com.procurebuddy.service.ChatExportService;
import com.procurebuddy.service.ChatService;
import jakarta.validation.Valid;
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
            @RequestParam("user") String user,
            @RequestParam(value = "page", required = false) Integer page,
            @RequestParam(value = "size", required = false) Integer size
    ) {
        return chatService.listChats(user, page, size);
    }

    @GetMapping({"/api/chats/{chatId}", "/chats/{chatId}"})
    public Map<String, Object> getChat(
            @PathVariable String chatId,
            @RequestParam("user") String user,
            @RequestParam(value = "page", required = false) Integer page,
            @RequestParam(value = "size", required = false) Integer size
    ) {
        return chatService.getChat(chatId, user, page, size);
    }

    @PostMapping({"/api/chats/{chatId}/message", "/chats/{chatId}/message"})
    public CompletableFuture<Map<String, Object>> sendMessage(@PathVariable String chatId, @Valid @RequestBody SendMessageRequest request) {
        return chatService.sendMessageAsync(chatId, request.getUser(), request.getMessage());
    }

    @PostMapping({"/api/chats/{chatId}/regenerate", "/chats/{chatId}/regenerate"})
    public CompletableFuture<Map<String, Object>> regenerateResponse(
            @PathVariable String chatId,
            @RequestParam("user") String user
    ) {
        return chatService.regenerateLastResponseAsync(chatId, user);
    }

    @PostMapping("/chat/regenerate")
    public CompletableFuture<Map<String, Object>> regenerateResponseCompat(
            @RequestParam("chatId") String chatId,
            @RequestParam("user") String user
    ) {
        return chatService.regenerateLastResponseAsync(chatId, user);
    }

    @DeleteMapping({"/api/chats/{chatId}", "/chats/{chatId}"})
    public Map<String, Object> deleteChat(
            @PathVariable String chatId,
            @RequestParam(value = "user", required = false) String user,
            @RequestParam(value = "userId", required = false) Long userId
    ) {
        return chatService.deleteChat(chatId, user, userId);
    }

    @PostMapping({"/api/chats/pin", "/chats/pin"})
    public Map<String, Object> pinChat(@RequestBody PinChatRequest request) {
        return chatService.pinChat(request.getChatId(), request.getPinned(), request.getUser(), request.getUserId());
    }

    @GetMapping(value = "/api/chats/{chatId}/export", produces = MediaType.APPLICATION_PDF_VALUE)
    public ResponseEntity<byte[]> exportChat(
            @PathVariable String chatId,
            @RequestParam("user") String user
    ) {
        ChatExportService.ChatExportResult result = chatExportService.exportChatPdf(chatId, user);
        return ResponseEntity.ok()
                .header(
                        HttpHeaders.CONTENT_DISPOSITION,
                        ContentDisposition.attachment().filename(result.filename()).build().toString()
                )
                .contentType(MediaType.APPLICATION_PDF)
                .body(result.content());
    }
}
