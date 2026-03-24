package com.procurebuddy.repository;

import com.procurebuddy.entity.ChatEntity;
import com.procurebuddy.entity.MessageEntity;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface MessageRepository extends JpaRepository<MessageEntity, Long> {

    List<MessageEntity> findAllByChatOrderByTimestampAscIdAsc(ChatEntity chat);

    Page<MessageEntity> findAllByChatOrderByTimestampAscIdAsc(ChatEntity chat, Pageable pageable);

    long countByChat(ChatEntity chat);

    void deleteAllByChat(ChatEntity chat);

    @Query("""
            select m.chat.id as chatId, count(m) as exchangeCount
            from MessageEntity m
            where m.chat.id in :chatIds
            group by m.chat.id
            """)
    List<ChatExchangeCountProjection> countAllByChatIds(@Param("chatIds") List<String> chatIds);

    interface ChatExchangeCountProjection {
        String getChatId();

        long getExchangeCount();
    }
}
