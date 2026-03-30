package com.procurebuddy.service;

import com.procurebuddy.entity.DocumentEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.DocumentRepository;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

@Service
@RequiredArgsConstructor
public class DocumentService {

    private final DocumentRepository documentRepository;

    @Transactional
    public Map<String, Object> storeDocuments(MultipartFile file, MultipartFile[] files) {
        return storeDocuments(mergeFiles(file, files));
    }

    @Transactional
    public Map<String, Object> storeDocuments(MultipartFile[] files) {
        return storeDocuments(mergeFiles(null, files));
    }

    @Transactional(readOnly = true)
    public Map<String, Object> searchDocuments(String query) {
        String normalizedQuery = normalizeQuery(query);
        List<LinkedHashMap<String, Object>> matches = documentRepository
                .search(normalizedQuery, PageRequest.of(0, 20))
                .stream()
                .map(document -> toSearchResult(document, normalizedQuery))
                .toList();

        return Map.of(
                "success", true,
                "query", normalizedQuery,
                "matches", matches,
                "count", matches.size()
        );
    }

    private LinkedHashMap<String, Object> storeSingleDocument(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Uploaded document must not be empty.");
        }

        String fileName = sanitizeFileName(file.getOriginalFilename());
        String extractedText = extractText(file);
        if (extractedText.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Unable to extract readable text from '" + fileName + "'.");
        }

        DocumentEntity document = documentRepository.findByFileNameIgnoreCase(fileName)
                .orElseGet(DocumentEntity::new);
        document.setFileName(fileName);
        document.setContentType(file.getContentType());
        document.setSizeBytes(file.getSize());
        document.setContent(extractedText);
        document.setUpdatedAt(LocalDateTime.now());
        document = documentRepository.save(document);

        LinkedHashMap<String, Object> item = new LinkedHashMap<>();
        item.put("id", document.getId());
        item.put("file_name", document.getFileName());
        item.put("content_type", document.getContentType());
        item.put("size_bytes", document.getSizeBytes());
        item.put("updated_at", document.getUpdatedAt());
        return item;
    }

    private List<MultipartFile> mergeFiles(MultipartFile file, MultipartFile[] files) {
        List<MultipartFile> merged = new ArrayList<>();
        if (file != null && !file.isEmpty()) {
            merged.add(file);
        }
        if (files != null) {
            for (MultipartFile current : files) {
                if (current != null && !current.isEmpty()) {
                    merged.add(current);
                }
            }
        }
        return merged;
    }

    private Map<String, Object> storeDocuments(List<MultipartFile> files) {
        if (files.isEmpty()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Select at least one document to upload.");
        }

        List<LinkedHashMap<String, Object>> storedDocuments = files.stream()
                .map(this::storeSingleDocument)
                .toList();

        return Map.of(
                "success", true,
                "documents", storedDocuments,
                "count", storedDocuments.size()
        );
    }

    private String sanitizeFileName(String originalFilename) {
        if (originalFilename == null || originalFilename.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Document file name is required.");
        }
        String fileName = Path.of(originalFilename).getFileName().toString().trim();
        if (fileName.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Document file name is required.");
        }
        return fileName;
    }

    private String extractText(MultipartFile file) {
        try {
            byte[] bytes = file.getBytes();
            String extension = extension(file.getOriginalFilename());
            return switch (extension) {
                case ".txt", ".md", ".csv", ".json", ".xml", ".html", ".log" ->
                        normalizeText(new String(bytes, StandardCharsets.UTF_8));
                case ".docx" -> extractDocxText(bytes);
                case ".pdf" -> extractPdfText(bytes);
                default -> {
                    String utf8Text = normalizeText(new String(bytes, StandardCharsets.UTF_8));
                    yield utf8Text.isBlank() ? extractPrintableText(bytes) : utf8Text;
                }
            };
        } catch (IOException ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to read uploaded document.");
        }
    }

    private String extension(String filename) {
        if (filename == null) {
            return "";
        }
        String normalized = filename.toLowerCase();
        int index = normalized.lastIndexOf('.');
        return index >= 0 ? normalized.substring(index) : "";
    }

    private String extractDocxText(byte[] bytes) {
        try (ZipInputStream input = new ZipInputStream(new ByteArrayInputStream(bytes))) {
            ZipEntry entry;
            while ((entry = input.getNextEntry()) != null) {
                if ("word/document.xml".equals(entry.getName())) {
                    String xml = new String(readAllBytes(input), StandardCharsets.UTF_8);
                    String text = xml
                            .replaceAll("</w:p>", "\n")
                            .replaceAll("<[^>]+>", " ")
                            .replace("&amp;", "&")
                            .replace("&lt;", "<")
                            .replace("&gt;", ">")
                            .replace("&quot;", "\"")
                            .replace("&apos;", "'");
                    return normalizeText(text);
                }
            }
        } catch (IOException ex) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Failed to extract text from DOCX document.");
        }
        return "";
    }

    private String extractPdfText(byte[] bytes) {
        String raw = new String(bytes, StandardCharsets.ISO_8859_1);
        StringBuilder extracted = new StringBuilder();
        StringBuilder current = new StringBuilder();
        boolean insideText = false;
        boolean escaped = false;

        for (int index = 0; index < raw.length(); index++) {
            char value = raw.charAt(index);
            if (!insideText) {
                if (value == '(') {
                    insideText = true;
                    current.setLength(0);
                }
                continue;
            }

            if (escaped) {
                current.append(switch (value) {
                    case 'n', 'r', 't' -> ' ';
                    default -> value;
                });
                escaped = false;
                continue;
            }

            if (value == '\\') {
                escaped = true;
                continue;
            }

            if (value == ')') {
                if (!current.isEmpty()) {
                    extracted.append(current).append('\n');
                }
                insideText = false;
                continue;
            }

            current.append(isReadablePdfChar(value) ? value : ' ');
        }

        String normalized = normalizeText(extracted.toString());
        return normalized.isBlank() ? extractPrintableText(bytes) : normalized;
    }

    private boolean isReadablePdfChar(char value) {
        return Character.isLetterOrDigit(value)
                || Character.isWhitespace(value)
                || ".,;:!?@#%&()[]{}+-_/\"'".indexOf(value) >= 0;
    }

    private String extractPrintableText(byte[] bytes) {
        StringBuilder text = new StringBuilder();
        StringBuilder chunk = new StringBuilder();
        for (byte value : bytes) {
            int normalized = value & 0xFF;
            char character = (char) normalized;
            if (character == '\n' || character == '\r' || character == '\t' || (character >= 32 && character <= 126)) {
                chunk.append(character);
            } else {
                flushChunk(chunk, text);
            }
        }
        flushChunk(chunk, text);
        return normalizeText(text.toString());
    }

    private void flushChunk(StringBuilder chunk, StringBuilder text) {
        if (chunk.length() >= 4) {
            text.append(chunk).append('\n');
        }
        chunk.setLength(0);
    }

    private byte[] readAllBytes(InputStream input) throws IOException {
        return input.readAllBytes();
    }

    private String normalizeText(String value) {
        if (value == null) {
            return "";
        }
        return value
                .replace('\u0000', ' ')
                .replaceAll("\\s+", " ")
                .trim();
    }

    private String normalizeQuery(String query) {
        if (query == null || query.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Search query is required.");
        }
        return query.trim();
    }

    private LinkedHashMap<String, Object> toSearchResult(DocumentEntity document, String query) {
        LinkedHashMap<String, Object> item = new LinkedHashMap<>();
        item.put("id", document.getId());
        item.put("file_name", document.getFileName());
        item.put("matched_content", buildExcerpt(document.getContent(), query));
        item.put("updated_at", document.getUpdatedAt());
        return item;
    }

    private String buildExcerpt(String content, String query) {
        if (content == null || content.isBlank()) {
            return "";
        }
        String loweredContent = content.toLowerCase();
        String loweredQuery = query.toLowerCase();
        int matchIndex = loweredContent.indexOf(loweredQuery);
        if (matchIndex < 0) {
            return content.substring(0, Math.min(content.length(), 240));
        }
        int start = Math.max(0, matchIndex - 80);
        int end = Math.min(content.length(), matchIndex + query.length() + 160);
        return content.substring(start, end).trim();
    }
}
