package com.procurebuddy.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Index;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.LocalDateTime;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(
        name = "prompt_stats",
        indexes = {
                @Index(name = "idx_prompt_stats_count_last_used", columnList = "prompt_count,last_used_at")
        }
)
public class PromptStatEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true, length = 1000)
    private String promptText;

    @Column(name = "prompt_count", nullable = false)
    private long count;

    @Column(name = "last_used_at", nullable = false)
    private LocalDateTime lastUsedAt;

    @PrePersist
    public void onCreate() {
        if (count <= 0) {
            count = 1;
        }
        if (lastUsedAt == null) {
            lastUsedAt = LocalDateTime.now();
        }
    }
}
