package com.procurebuddy.service;

import com.procurebuddy.config.ProcureBuddyProperties;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.util.UserResolver;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

@Service
@RequiredArgsConstructor
public class AdminService {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(".pdf", ".docx", ".txt");

    private final ProcureBuddyProperties properties;
    private final AuthService authService;
    private final PythonBridgeService pythonBridgeService;
    private final ReentrantLock processLock = new ReentrantLock();
    private final Map<String, Object> processState = new HashMap<>();

    @Transactional(readOnly = true)
    public Map<String, Object> listDocuments(String email) {
        requireAdmin(email);
        try {
            Path dataDir = resolveDataDir();
            if (!Files.exists(dataDir)) {
                Files.createDirectories(dataDir);
            }
            List<Map<String, Object>> documents = Files.list(dataDir)
                    .filter(Files::isRegularFile)
                    .filter(path -> SUPPORTED_EXTENSIONS.contains(extension(path)))
                    .sorted(Comparator.comparing(path -> path.getFileName().toString().toLowerCase()))
                    .map(this::toDocumentInfo)
                    .toList();
            return Map.of("success", true, "documents", documents, "count", documents.size());
        } catch (IOException ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to list documents.");
        }
    }

    @Transactional(readOnly = true)
    public Map<String, Object> status(String email) {
        requireAdmin(email);
        LinkedHashMap<String, Object> response = new LinkedHashMap<>();
        response.put("success", true);
        response.putAll(currentState());
        return response;
    }

    public Map<String, Object> uploadDocuments(String email, MultipartFile[] files) {
        requireAdmin(email);
        if (files == null || files.length == 0) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Select at least one document to upload.");
        }

        List<String> uploaded = new ArrayList<>();
        try {
            Path dataDir = resolveDataDir();
            Files.createDirectories(dataDir);
            for (MultipartFile file : files) {
                String filename = Path.of(file.getOriginalFilename() == null ? "" : file.getOriginalFilename()).getFileName().toString();
                String extension = extension(Path.of(filename));
                if (filename.isBlank() || !SUPPORTED_EXTENSIONS.contains(extension)) {
                    throw new ApiException(
                            HttpStatus.BAD_REQUEST,
                            "Unsupported file '" + file.getOriginalFilename() + "'. Allowed: " + String.join(", ", SUPPORTED_EXTENSIONS.stream().sorted().toList())
                    );
                }
                Files.copy(file.getInputStream(), dataDir.resolve(filename), StandardCopyOption.REPLACE_EXISTING);
                uploaded.add(filename);
            }
        } catch (IOException ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to store uploaded documents.");
        }

        Map<String, Object> result = runProcessingCycle("upload");
        LinkedHashMap<String, Object> response = new LinkedHashMap<>();
        response.put("success", true);
        response.put("message", "Documents uploaded and knowledge base refreshed successfully.");
        response.put("uploaded", uploaded);
        response.putAll(result);
        return response;
    }

    public Map<String, Object> reindexDocuments(String email) {
        requireAdmin(email);
        Map<String, Object> result = runProcessingCycle("reindex");
        LinkedHashMap<String, Object> response = new LinkedHashMap<>();
        response.put("success", true);
        response.put("message", "Knowledge base reindexed successfully.");
        response.putAll(result);
        return response;
    }

    public boolean isBusy() {
        return Boolean.TRUE.equals(processState.get("busy"));
    }

    private void requireAdmin(String email) {
        if (!authService.isAdminEmail(UserResolver.normalizeEmail(email))) {
            throw new ApiException(HttpStatus.FORBIDDEN, "Admin access is restricted to the configured admin account.");
        }
    }

    private Map<String, Object> runProcessingCycle(String trigger) {
        if (!processLock.tryLock()) {
            throw new ApiException(HttpStatus.CONFLICT, "Document processing is already running.");
        }

        updateState(Map.of(
                "busy", true,
                "stage", trigger + ": preparing",
                "started_at", LocalDateTime.now()
        ));
        updateState(withNullableEntries("finished_at", null, "last_error", null));

        try {
            updateState(Map.of("stage", trigger + ": OCR, chunking, and vector refresh"));
            Map<String, Object> result = pythonBridgeService.reindexKnowledgeBase();
            LocalDateTime finishedAt = LocalDateTime.now();
            updateState(Map.of(
                    "busy", false,
                    "stage", "idle",
                    "finished_at", finishedAt
            ));
            LinkedHashMap<String, Object> lastResult = new LinkedHashMap<>(result);
            lastResult.put("trigger", trigger);
            lastResult.put("finished_at", finishedAt);
            updateState(withNullableEntries("last_result", lastResult, "last_error", null));
            return result;
        } catch (RuntimeException ex) {
            LinkedHashMap<String, Object> values = new LinkedHashMap<>();
            values.put("busy", false);
            values.put("stage", "idle");
            values.put("finished_at", LocalDateTime.now());
            values.put("last_error", ex.getMessage());
            updateState(values);
            throw ex;
        } finally {
            processLock.unlock();
        }
    }

    private synchronized void updateState(Map<String, Object> values) {
        processState.putAll(values);
    }

    private Map<String, Object> withNullableEntries(String firstKey, Object firstValue, String secondKey, Object secondValue) {
        LinkedHashMap<String, Object> values = new LinkedHashMap<>();
        values.put(firstKey, firstValue);
        values.put(secondKey, secondValue);
        return values;
    }

    private synchronized Map<String, Object> currentState() {
        LinkedHashMap<String, Object> state = new LinkedHashMap<>();
        state.put("busy", processState.getOrDefault("busy", false));
        state.put("stage", processState.getOrDefault("stage", "idle"));
        state.put("started_at", processState.get("started_at"));
        state.put("finished_at", processState.get("finished_at"));
        state.put("last_result", processState.get("last_result"));
        state.put("last_error", processState.get("last_error"));
        return state;
    }

    private Path resolveDataDir() {
        return Path.of(properties.getDataDir()).toAbsolutePath().normalize();
    }

    private String extension(Path path) {
        String filename = path.getFileName().toString().toLowerCase();
        int index = filename.lastIndexOf('.');
        return index >= 0 ? filename.substring(index) : "";
    }

    private Map<String, Object> toDocumentInfo(Path path) {
        try {
            long size = Files.size(path);
            Instant modified = Files.getLastModifiedTime(path).toInstant();
            return Map.of(
                    "name", path.getFileName().toString(),
                    "type", extension(path).replace(".", ""),
                    "size_bytes", size,
                    "size_label", formatSize(size),
                    "updated_at", LocalDateTime.ofInstant(modified, ZoneId.systemDefault())
            );
        } catch (IOException ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to inspect document metadata.");
        }
    }

    private String formatSize(long bytes) {
        String[] units = {"B", "KB", "MB", "GB"};
        double value = bytes;
        for (String unit : units) {
            if (value < 1024 || "GB".equals(unit)) {
                if ("B".equals(unit)) {
                    return (long) value + " " + unit;
                }
                return String.format("%.1f %s", value, unit);
            }
            value /= 1024;
        }
        return bytes + " B";
    }
}
