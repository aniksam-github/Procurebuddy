package com.procurebuddy.controller;

import com.procurebuddy.dto.request.FeedbackRequest;
import com.procurebuddy.service.FeedbackService;
import jakarta.validation.Valid;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class FeedbackController {

    private final FeedbackService feedbackService;

    @PostMapping({"/api/feedback", "/feedback"})
    public Map<String, Object> submitFeedback(@Valid @RequestBody FeedbackRequest request) {
        return feedbackService.submitFeedback(
                request.getUser(),
                request.getMessageId(),
                request.getType(),
                request.getChatId()
        );
    }
}
