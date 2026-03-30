package com.procurebuddy.service;

import com.procurebuddy.entity.FeedbackEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.FeedbackRepository;
import com.procurebuddy.util.UserResolver;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class FeedbackService {

    private final FeedbackRepository feedbackRepository;

    @Transactional
    public Map<String, Object> submitFeedback(String user, String messageId, String type, String chatId) {
        if (user == null || user.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "User email is required.");
        }
        if (messageId == null || messageId.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Message id is required.");
        }

        String normalizedEmail = UserResolver.normalizeEmail(user);
        String normalizedType = normalizeType(type);
        FeedbackEntity feedback = feedbackRepository.findByUserEmailIgnoreCaseAndMessageId(normalizedEmail, messageId)
                .orElseGet(FeedbackEntity::new);
        feedback.setUserEmail(normalizedEmail);
        feedback.setMessageId(messageId);
        feedback.setChatId(chatId);
        feedback.setType(normalizedType);
        feedback.setTimestamp(LocalDateTime.now());
        feedbackRepository.save(feedback);

        LinkedHashMap<String, Object> item = new LinkedHashMap<>();
        item.put("message_id", feedback.getMessageId());
        item.put("type", feedback.getType());
        item.put("timestamp", feedback.getTimestamp());
        return Map.of("success", true, "feedback", item);
    }

    private String normalizeType(String type) {
        if (type == null) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Feedback type is required.");
        }
        return switch (type.strip().toLowerCase()) {
            case "up", "like", "thumbs_up" -> "up";
            case "down", "dislike", "thumbs_down" -> "down";
            default -> throw new ApiException(HttpStatus.BAD_REQUEST, "Feedback type must be 'up' or 'down'.");
        };
    }
}
