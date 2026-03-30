package com.procurebuddy.repository;

import com.procurebuddy.entity.ChatEntity;
import com.procurebuddy.entity.MessageRevisionEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MessageRevisionRepository extends JpaRepository<MessageRevisionEntity, Long> {

    void deleteAllByMessageChat(ChatEntity chat);
}
