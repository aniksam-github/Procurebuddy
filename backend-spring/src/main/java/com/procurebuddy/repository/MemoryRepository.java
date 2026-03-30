package com.procurebuddy.repository;

import com.procurebuddy.entity.MemoryEntity;
import com.procurebuddy.entity.UserEntity;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MemoryRepository extends JpaRepository<MemoryEntity, Long> {

    Optional<MemoryEntity> findByUserAndMemoryKeyIgnoreCase(UserEntity user, String memoryKey);

    List<MemoryEntity> findAllByUserOrderByUpdatedAtDesc(UserEntity user);
}
