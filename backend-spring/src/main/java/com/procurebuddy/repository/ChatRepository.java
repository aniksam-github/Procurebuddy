package com.procurebuddy.repository;

import com.procurebuddy.entity.ChatEntity;
import com.procurebuddy.entity.UserEntity;
import java.util.List;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ChatRepository extends JpaRepository<ChatEntity, String> {

    List<ChatEntity> findAllByUserOrderByPinnedDescUpdatedAtDesc(UserEntity user);

    Page<ChatEntity> findAllByUserOrderByPinnedDescUpdatedAtDesc(UserEntity user, Pageable pageable);

    Optional<ChatEntity> findByIdAndUser(String id, UserEntity user);
}
