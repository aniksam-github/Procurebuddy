package com.procurebuddy.repository;

import com.procurebuddy.entity.DocumentChunkEntity;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DocumentChunkRepository extends JpaRepository<DocumentChunkEntity, Long> {

    List<DocumentChunkEntity> findAllByOrderByIdAsc();
}
