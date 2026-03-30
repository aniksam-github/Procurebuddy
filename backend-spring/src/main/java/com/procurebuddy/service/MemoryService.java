package com.procurebuddy.service;

import com.procurebuddy.entity.MemoryEntity;
import com.procurebuddy.entity.UserEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.MemoryRepository;
import com.procurebuddy.util.UserResolver;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class MemoryService {

    private final MemoryRepository memoryRepository;
    private final UserResolver userResolver;

    @Transactional
    public Map<String, Object> upsertMemory(String email, Long userId, String key, String value) {
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        String normalizedKey = normalizeKey(key);
        String normalizedValue = normalizeValue(value);

        MemoryEntity memory = memoryRepository.findByUserAndMemoryKeyIgnoreCase(user, normalizedKey)
                .orElseGet(MemoryEntity::new);
        memory.setUser(user);
        memory.setMemoryKey(normalizedKey);
        memory.setMemoryValue(normalizedValue);
        memory.setUpdatedAt(LocalDateTime.now());
        memory = memoryRepository.save(memory);

        return Map.of("success", true, "memory", toResponse(memory));
    }

    @Transactional(readOnly = true)
    public Map<String, Object> getMemory(String email, Long userId, String key) {
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        if (key != null && !key.isBlank()) {
            MemoryEntity memory = memoryRepository.findByUserAndMemoryKeyIgnoreCase(user, key.trim())
                    .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "Memory entry not found."));
            return Map.of("success", true, "memory", toResponse(memory));
        }

        List<LinkedHashMap<String, Object>> items = memoryRepository.findAllByUserOrderByUpdatedAtDesc(user)
                .stream()
                .map(this::toResponse)
                .toList();
        return Map.of("success", true, "items", items, "count", items.size());
    }

    private String normalizeKey(String key) {
        if (key == null || key.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Memory key is required.");
        }
        String normalized = key.trim().replaceAll("\\s+", " ");
        if (normalized.length() > 100) {
            normalized = normalized.substring(0, 100).trim();
        }
        return normalized;
    }

    private String normalizeValue(String value) {
        if (value == null || value.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Memory value is required.");
        }
        return value.trim();
    }

    private LinkedHashMap<String, Object> toResponse(MemoryEntity memory) {
        LinkedHashMap<String, Object> item = new LinkedHashMap<>();
        item.put("id", memory.getId());
        item.put("user_id", memory.getUser().getId());
        item.put("key", memory.getMemoryKey());
        item.put("value", memory.getMemoryValue());
        item.put("updated_at", memory.getUpdatedAt());
        return item;
    }
}
