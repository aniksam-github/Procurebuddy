package com.procurebuddy.service;

import com.procurebuddy.dto.response.ChatMessageResponse;
import com.procurebuddy.dto.response.ChatSummaryResponse;
import com.procurebuddy.entity.ChatEntity;
import com.procurebuddy.entity.FolderEntity;
import com.procurebuddy.entity.MessageEntity;
import com.procurebuddy.entity.UserEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.ChatRepository;
import com.procurebuddy.repository.FolderRepository;
import com.procurebuddy.repository.MessageRepository;
import com.procurebuddy.util.UserResolver;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@RequiredArgsConstructor
public class ChatService {

    private static final String AI_FALLBACK_MESSAGE =
            "I could not process that request right now because the chatbot backend is not fully configured. Please check the server logs.";

    private final ChatRepository chatRepository;
    private final MessageRepository messageRepository;
    private final FolderRepository folderRepository;
    private final UserResolver userResolver;
    private final PythonBridgeService pythonBridgeService;
    private final AdminService adminService;

    @Cacheable(cacheNames = "chatLists")
    @Transactional(readOnly = true)
    public Map<String, Object> listChats(String email, Integer page, Integer size) {
        UserEntity user = userResolver.requireByEmail(email);
        Pageable pageable = buildPageable(page, size, Sort.by(Sort.Order.desc("pinned"), Sort.Order.desc("updatedAt")));
        List<ChatEntity> chatEntities;
        if (pageable.isPaged()) {
            Page<ChatEntity> chatPage = chatRepository.findAllByUserOrderByPinnedDescUpdatedAtDesc(user, pageable);
            chatEntities = chatPage.getContent();
        } else {
            chatEntities = chatRepository.findAllByUserOrderByPinnedDescUpdatedAtDesc(user);
        }
        Map<String, Long> exchangeCounts = loadExchangeCounts(chatEntities);
        List<ChatSummaryResponse> chats = chatEntities.stream()
                .map(chat -> toSummary(chat, exchangeCounts.getOrDefault(chat.getId(), 0L)))
                .toList();

        return Map.of(
                "chat_ids", chats.stream().map(ChatSummaryResponse::chatId).toList(),
                "chats", chats
        );
    }

    @Cacheable(cacheNames = "chatMessages")
    @Transactional(readOnly = true)
    public Map<String, Object> getChat(String chatId, String email, Integer page, Integer size) {
        UserEntity user = userResolver.requireByEmail(email);
        return chatRepository.findByIdAndUser(chatId, user)
                .map(chat -> Map.of(
                        "chat_id", chat.getId(),
                        "title", (chat.getTitle() == null || chat.getTitle().isBlank()) ? "New Chat" : chat.getTitle(),
                        "messages", expandMessages(chat, page, size)
                ))
                .orElseGet(() -> Map.of(
                        "chat_id", chatId,
                        "title", "New Chat",
                        "messages", List.of()
                ));
    }

    @Async("aiTaskExecutor")
    @CacheEvict(cacheNames = {"chatLists", "chatMessages"}, allEntries = true)
    @Transactional
    public CompletableFuture<Map<String, Object>> sendMessageAsync(String chatId, String email, String text) {
        if (adminService.isBusy()) {
            throw new ApiException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Knowledge base update in progress. Chat is temporarily paused until processing completes."
            );
        }
        if (email == null || email.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "User email is required.");
        }
        if (text == null || text.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Message is required.");
        }

        UserEntity user = userResolver.requireByEmail(email);
        ChatEntity chat = chatRepository.findByIdAndUser(chatId, user).orElseGet(() -> createChat(chatId, user));

        List<ChatMessageResponse> history = expandMessages(chat, null, null);
        String reply;
        try {
            reply = pythonBridgeService.askQuestion(text.trim(), history);
        } catch (Exception ex) {
            log.error("Chat reply generation failed", ex);
            reply = AI_FALLBACK_MESSAGE;
        }

        MessageEntity exchange = new MessageEntity();
        exchange.setChat(chat);
        exchange.setMessage(text.trim());
        exchange.setResponse(reply);
        exchange.setTimestamp(LocalDateTime.now());
        messageRepository.save(exchange);

        updateChatMetadata(chat, exchange);
        chatRepository.save(chat);

        List<ChatMessageResponse> allMessages = expandMessages(chat, null, null);
        LinkedHashMap<String, Object> response = new LinkedHashMap<>();
        response.put("reply", reply);
        response.put("chat", toSummary(chat, messageRepository.countByChat(chat)));
        response.put("messages", allMessages);
        return CompletableFuture.completedFuture(response);
    }

    @CacheEvict(cacheNames = {"chatLists", "chatMessages"}, allEntries = true)
    @Transactional
    public Map<String, Object> deleteChat(String chatId, String email, Long userId) {
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        ChatEntity chat = chatRepository.findByIdAndUser(chatId, user)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "Chat not found."));
        messageRepository.deleteAllByChat(chat);
        chatRepository.delete(chat);
        return Map.of("success", true, "message", "Chat deleted.");
    }

    @CacheEvict(cacheNames = {"chatLists", "chatMessages"}, allEntries = true)
    @Transactional
    public Map<String, Object> pinChat(String chatId, Boolean pinned, String email, Long userId) {
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        ChatEntity chat = chatRepository.findByIdAndUser(chatId, user)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "Chat not found."));
        chat.setPinned(Boolean.TRUE.equals(pinned));
        chatRepository.save(chat);
        return Map.of("success", true, "chat", toSummary(chat, messageRepository.countByChat(chat)));
    }

    @CacheEvict(cacheNames = {"chatLists", "chatMessages"}, allEntries = true)
    @Transactional
    public Map<String, Object> moveChatToFolder(String chatId, String folderId, String email, Long userId) {
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        ChatEntity chat = chatRepository.findByIdAndUser(chatId, user)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "Chat not found."));

        FolderEntity folder = null;
        if (folderId != null && !folderId.isBlank()) {
            folder = folderRepository.findByIdAndUser(folderId, user)
                    .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "Folder not found."));
        }
        chat.setFolder(folder);
        chatRepository.save(chat);
        return Map.of("success", true, "chat", toSummary(chat, messageRepository.countByChat(chat)));
    }

    @Transactional(readOnly = true)
    public List<ChatMessageResponse> expandMessages(ChatEntity chat, Integer page, Integer size) {
        List<ChatMessageResponse> messages = new ArrayList<>();
        List<MessageEntity> exchanges;
        Pageable pageable = buildPageable(page, size, Sort.by(Sort.Order.asc("timestamp"), Sort.Order.asc("id")));
        if (pageable.isPaged()) {
            exchanges = messageRepository.findAllByChatOrderByTimestampAscIdAsc(chat, pageable).getContent();
        } else {
            exchanges = messageRepository.findAllByChatOrderByTimestampAscIdAsc(chat);
        }
        for (MessageEntity exchange : exchanges) {
            messages.add(ChatMessageResponse.builder()
                    .role("user")
                    .content(exchange.getMessage())
                    .timestamp(exchange.getTimestamp())
                    .build());
            messages.add(ChatMessageResponse.builder()
                    .role("assistant")
                    .content(exchange.getResponse())
                    .timestamp(exchange.getTimestamp())
                    .build());
        }
        return messages;
    }

    @Transactional(readOnly = true)
    public ChatSummaryResponse toSummary(ChatEntity chat, long exchangeCount) {
        long messageCount = exchangeCount * 2L;
        return ChatSummaryResponse.builder()
                .chatId(chat.getId())
                .title((chat.getTitle() == null || chat.getTitle().isBlank()) ? "New Chat" : chat.getTitle())
                .preview(chat.getPreview() == null ? "" : chat.getPreview())
                .messageCount(messageCount)
                .updatedAt(chat.getUpdatedAt())
                .isPinned(chat.isPinned())
                .folderId(chat.getFolder() == null ? null : chat.getFolder().getId())
                .build();
    }

    private ChatEntity createChat(String chatId, UserEntity user) {
        ChatEntity chat = new ChatEntity();
        chat.setId(chatId);
        chat.setUser(user);
        chat.setTitle("New Chat");
        chat.setPreview("Start a new procurement query.");
        chat.setPinned(false);
        return chatRepository.save(chat);
    }

    private void updateChatMetadata(ChatEntity chat, MessageEntity latestExchange) {
        if (chat.getTitle() == null || chat.getTitle().isBlank() || "New Chat".equals(chat.getTitle())) {
            chat.setTitle(trimForSummary(latestExchange.getMessage(), 60));
        }
        chat.setPreview(trimForSummary(
                latestExchange.getResponse() != null && !latestExchange.getResponse().isBlank()
                        ? latestExchange.getResponse()
                        : latestExchange.getMessage(),
                120
        ));
        chat.setUpdatedAt(latestExchange.getTimestamp() == null ? LocalDateTime.now() : latestExchange.getTimestamp());
    }

    private Map<String, Long> loadExchangeCounts(List<ChatEntity> chats) {
        if (chats.isEmpty()) {
            return Map.of();
        }
        Map<String, Long> counts = new HashMap<>();
        List<String> ids = chats.stream().map(ChatEntity::getId).toList();
        messageRepository.countAllByChatIds(ids).forEach(item -> counts.put(item.getChatId(), item.getExchangeCount()));
        return counts;
    }

    private Pageable buildPageable(Integer page, Integer size, Sort sort) {
        if (page == null && size == null) {
            return Pageable.unpaged();
        }
        int resolvedPage = page == null || page < 0 ? 0 : page;
        int resolvedSize = size == null || size <= 0 ? 20 : Math.min(size, 200);
        return PageRequest.of(resolvedPage, resolvedSize, sort);
    }

    private String trimForSummary(String value, int maxLength) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String normalized = value.strip().replace("\n", " ");
        return normalized.substring(0, Math.min(normalized.length(), maxLength));
    }
}
