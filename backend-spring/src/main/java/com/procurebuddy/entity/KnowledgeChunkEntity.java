package com.procurebuddy.entity;

import com.procurebuddy.persistence.FloatArrayStringConverter;
import jakarta.persistence.Column;
import jakarta.persistence.Convert;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;
import java.time.LocalDateTime;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(
        name = "knowledge_chunks",
        uniqueConstraints = {
                @UniqueConstraint(name = "uk_knowledge_chunks_document_chunk", columnNames = {"document_id", "chunk_index"})
        },
        indexes = {
                @Index(name = "idx_knowledge_chunks_document", columnList = "document_id"),
                @Index(name = "idx_knowledge_chunks_source_file", columnList = "source_file_name"),
                @Index(name = "idx_knowledge_chunks_updated_at", columnList = "updated_at")
        }
)
public class KnowledgeChunkEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "document_id", nullable = false)
    private DocumentEntity document;

    @Column(name = "source_file_name", nullable = false)
    private String sourceFileName;

    @Column(name = "chunk_index", nullable = false)
    private int chunkIndex;

    @Column(name = "token_count", nullable = false)
    private int tokenCount;

    @Column(name = "embedding_model", nullable = false, length = 120)
    private String embeddingModel;

    @Column(name = "embedding_vector", nullable = false, columnDefinition = "TEXT")
    @Convert(converter = FloatArrayStringConverter.class)
    private float[] embeddingVector;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @PrePersist
    public void onCreate() {
        LocalDateTime now = LocalDateTime.now();
        if (createdAt == null) {
            createdAt = now;
        }
        if (updatedAt == null) {
            updatedAt = now;
        }
    }

    @PreUpdate
    public void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
