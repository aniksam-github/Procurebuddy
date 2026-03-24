package com.procurebuddy.repository;

import com.procurebuddy.entity.PendingOtpEntity;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PendingOtpRepository extends JpaRepository<PendingOtpEntity, Long> {

    Optional<PendingOtpEntity> findByEmailIgnoreCase(String email);
}
