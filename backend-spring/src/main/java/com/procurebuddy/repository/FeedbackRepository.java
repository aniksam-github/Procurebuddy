package com.procurebuddy.repository;

import com.procurebuddy.entity.FeedbackEntity;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface FeedbackRepository extends JpaRepository<FeedbackEntity, Long> {

    Optional<FeedbackEntity> findByUserEmailIgnoreCaseAndMessageId(String userEmail, String messageId);

    void deleteAllByChatId(String chatId);
}
