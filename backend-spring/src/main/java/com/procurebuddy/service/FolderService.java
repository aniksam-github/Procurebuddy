package com.procurebuddy.service;

import com.procurebuddy.dto.response.FolderResponse;
import com.procurebuddy.entity.ChatEntity;
import com.procurebuddy.entity.FolderEntity;
import com.procurebuddy.entity.UserEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.ChatRepository;
import com.procurebuddy.repository.FolderRepository;
import com.procurebuddy.util.UserResolver;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class FolderService {

    private final FolderRepository folderRepository;
    private final ChatRepository chatRepository;
    private final UserResolver userResolver;

    @CacheEvict(cacheNames = {"chatLists", "chatMessages"}, allEntries = true)
    @Transactional
    public Map<String, Object> createFolder(String email, Long userId, String name) {
        if (name == null || name.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Folder name is required.");
        }
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        FolderEntity folder = new FolderEntity();
        folder.setUser(user);
        folder.setName(name.trim());
        folderRepository.save(folder);
        return Map.of(
                "success", true,
                "folder", toResponse(folder)
        );
    }

    @Transactional(readOnly = true)
    public Map<String, Object> listFolders(String email, Long userId) {
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        List<FolderResponse> folders = folderRepository.findAllByUserOrderByCreatedAtAsc(user)
                .stream()
                .map(this::toResponse)
                .toList();
        return Map.of("success", true, "folders", folders, "count", folders.size());
    }

    @CacheEvict(cacheNames = {"chatLists", "chatMessages"}, allEntries = true)
    @Transactional
    public Map<String, Object> deleteFolder(String folderId, String email, Long userId) {
        UserEntity user = userResolver.requireByIdentifier(email, userId);
        FolderEntity folder = folderRepository.findByIdAndUser(folderId, user)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "Folder not found."));
        List<ChatEntity> chats = chatRepository.findAllByUserOrderByPinnedDescUpdatedAtDesc(user)
                .stream()
                .filter(chat -> chat.getFolder() != null && folder.getId().equals(chat.getFolder().getId()))
                .toList();
        chats.forEach(chat -> chat.setFolder(null));
        chatRepository.saveAll(chats);
        folderRepository.delete(folder);
        return Map.of("success", true, "message", "Folder deleted.");
    }

    private FolderResponse toResponse(FolderEntity folder) {
        return FolderResponse.builder()
                .id(folder.getId())
                .name(folder.getName())
                .createdAt(folder.getCreatedAt())
                .build();
    }
}
