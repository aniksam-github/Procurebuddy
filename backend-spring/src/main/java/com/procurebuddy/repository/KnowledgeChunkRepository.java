package com.procurebuddy.repository;

import com.procurebuddy.entity.DocumentEntity;
import com.procurebuddy.entity.KnowledgeChunkEntity;
import java.util.Collection;
import org.springframework.data.jpa.repository.JpaRepository;

public interface KnowledgeChunkRepository extends JpaRepository<KnowledgeChunkEntity, Long> {

    long deleteAllByDocument(DocumentEntity document);

    long deleteAllByDocument_IdIn(Collection<Long> documentIds);
}
