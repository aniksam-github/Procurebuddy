package com.procurebuddy.repository;

import com.procurebuddy.entity.PromptStatEntity;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PromptStatRepository extends JpaRepository<PromptStatEntity, Long> {

    Optional<PromptStatEntity> findByPromptText(String promptText);

    List<PromptStatEntity> findTop20ByOrderByCountDescLastUsedAtDesc();
}
