package com.procurebuddy.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import jakarta.persistence.Index;
import java.time.LocalDateTime;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
@Table(
        name = "users",
        indexes = {
                @Index(name = "idx_users_email", columnList = "email")
        }
)
public class UserEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String email;

    @Column(nullable = false)
    private String displayName;

    @Column(nullable = false)
    private String username;

    @Column(columnDefinition = "TEXT")
    private String avatarBase64;

    @Column(nullable = false)
    private String passwordHash;

    @Column(nullable = false)
    private boolean mustChange;

    @Column(nullable = false)
    private boolean totpEnabled;

    private String totpSecret;

    private String pendingTotpSecret;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @PrePersist
    public void onCreate() {
        if (displayName == null || displayName.isBlank()) {
            displayName = email;
        }
        if (username == null || username.isBlank()) {
            username = email == null ? "user" : email.split("@")[0];
        }
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }
}
