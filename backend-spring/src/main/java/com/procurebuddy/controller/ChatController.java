package com.procurebuddy.controller;

import com.procurebuddy.dto.request.PinChatRequest;
import com.procurebuddy.dto.request.SendMessageRequest;
import com.procurebuddy.service.ChatService;
import jakarta.validation.Valid;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
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
@RequestMapping
@RequiredArgsConstructor
public class ChatController {

    private final ChatService chatService;

    @GetMapping("/api/chats")
    public Map<String, Object> listChats(
            @RequestParam("user") String user,
            @RequestParam(value = "page", required = false) Integer page,
            @RequestParam(value = "size", required = false) Integer size
    ) {
        return chatService.listChats(user, page, size);
    }

    @GetMapping("/api/chats/{chatId}")
    public Map<String, Object> getChat(
            @PathVariable String chatId,
            @RequestParam("user") String user,
            @RequestParam(value = "page", required = false) Integer page,
            @RequestParam(value = "size", required = false) Integer size
    ) {
        return chatService.getChat(chatId, user, page, size);
    }

    @PostMapping("/api/chats/{chatId}/message")
    public CompletableFuture<Map<String, Object>> sendMessage(@PathVariable String chatId, @Valid @RequestBody SendMessageRequest request) {
        return chatService.sendMessageAsync(chatId, request.getUser(), request.getMessage());
    }

    @DeleteMapping("/api/chats/{chatId}")
    public Map<String, Object> deleteChat(
            @PathVariable String chatId,
            @RequestParam(value = "user", required = false) String user,
            @RequestParam(value = "userId", required = false) Long userId
    ) {
        return chatService.deleteChat(chatId, user, userId);
    }

    @PostMapping("/api/chats/pin")
    public Map<String, Object> pinChat(@RequestBody PinChatRequest request) {
        return chatService.pinChat(request.getChatId(), request.getPinned(), request.getUser(), request.getUserId());
    }
}
