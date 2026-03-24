package com.procurebuddy.repository;

import com.procurebuddy.entity.FolderEntity;
import com.procurebuddy.entity.UserEntity;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface FolderRepository extends JpaRepository<FolderEntity, String> {

    List<FolderEntity> findAllByUserOrderByCreatedAtAsc(UserEntity user);

    Optional<FolderEntity> findByIdAndUser(String id, UserEntity user);
}
