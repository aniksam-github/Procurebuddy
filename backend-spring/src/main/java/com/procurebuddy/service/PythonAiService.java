package com.procurebuddy.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.procurebuddy.config.ProcureBuddyProperties;
import com.procurebuddy.exception.ApiException;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Slf4j
@Service
public class PythonAiService {

    private static final String AI_UNAVAILABLE_MESSAGE = "AI service temporarily unavailable. Please try again.";

    private final String baseUrl;
    private final HttpClient httpClient;
    private final Duration readTimeout;
    private final ObjectMapper objectMapper;

    public PythonAiService(
            ProcureBuddyProperties properties,
            ObjectMapper objectMapper
    ) {
        this.baseUrl = normalizeBaseUrl(properties.getPythonService().getBaseUrl());
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(Math.max(1, properties.getPythonService().getConnectTimeoutSeconds())))
                .build();
        this.readTimeout = Duration.ofSeconds(Math.max(15, properties.getPythonService().getReadTimeoutSeconds()));
        this.objectMapper = objectMapper;
    }

    public AiChatResult chat(String message, String user, FeedbackService.FeedbackAwareChatContext feedbackContext) {
        String normalizedMessage = normalizeMessage(message);
        String normalizedUser = normalizeUser(user);
        log.info("Calling Python AI service for user='{}' messageLength={}", normalizedUser, normalizedMessage.length());
        try {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("message", normalizedMessage);
            payload.put("user", normalizedUser);
            payload.put("bypass_cache", feedbackContext != null && feedbackContext.bypassCache());
            payload.put("blocked_chunk_ids", feedbackContext == null ? List.of() : feedbackContext.blockedChunkIds());
            payload.put("blocked_response_hashes", feedbackContext == null ? List.of() : feedbackContext.blockedResponseHashes());
            AiChatResult response = exchange(
                    "/chat",
                    payload,
                    AiChatResult.class
            );
            if (response == null || response.response() == null || response.response().isBlank()) {
                log.error("Python AI service returned an empty response for user='{}'", normalizedUser);
                return unavailableResult();
            }
            log.info("Python AI service returned responseLength={}", response.response().length());
            return response.normalized();
        } catch (ApiException ex) {
            log.error("Python AI service unavailable for user='{}': {}", normalizedUser, ex.getMessage());
            return unavailableResult();
        } catch (Exception ex) {
            log.error("Unexpected Python AI service failure for user='{}'", normalizedUser, ex);
            return unavailableResult();
        }
    }

    public SearchResponse searchKnowledgeBase(String query) {
        String normalizedQuery = normalizeMessage(query);
        log.info("Calling Python search endpoint queryLength={}", normalizedQuery.length());
        SearchResponse response = exchange(
                "/search",
                Map.of("query", normalizedQuery),
                SearchResponse.class
        );
        if (response == null) {
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Python AI service returned an invalid search response.");
        }
        log.info("Python search returned {} matches", response.count() == null ? 0 : response.count());
        return response;
    }

    public Map<String, Object> reloadKnowledgeBase() {
        log.info("Calling Python reload endpoint at {}", endpoint("/reload"));
        Map<String, Object> response = exchange(
                "/reload",
                Map.of(),
                new TypeReference<>() { }
        );
        if (response == null || response.isEmpty()) {
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Python AI service returned an invalid reload response.");
        }
        log.info("Python reload completed successfully");
        return response;
    }

    public Map<String, Object> knowledgeBaseStatus() {
        log.info("Calling Python health endpoint at {}", endpoint("/health"));
        Map<String, Object> response = exchangeGet(
                "/health",
                new TypeReference<>() { }
        );
        if (response == null || response.isEmpty()) {
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Python AI service returned an invalid health response.");
        }
        return response;
    }

    private String normalizeMessage(String value) {
        if (value == null || value.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "Message is required.");
        }
        return value.trim();
    }

    private String normalizeUser(String value) {
        if (value == null || value.isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "User is required.");
        }
        return value.trim();
    }

    private String endpoint(String path) {
        return baseUrl + path;
    }

    private String endpoint(String candidateBaseUrl, String path) {
        return candidateBaseUrl + path;
    }

    private AiChatResult unavailableResult() {
        return new AiChatResult(AI_UNAVAILABLE_MESSAGE, "resp_unavailable", "resp_unavailable", List.of());
    }

    private <T> T exchange(String path, Object payload, Class<T> responseType) {
        return exchange(path, payload, body -> objectMapper.readValue(body, responseType));
    }

    private <T> T exchange(String path, Object payload, TypeReference<T> responseType) {
        return exchange(path, payload, body -> objectMapper.readValue(body, responseType));
    }

    private <T> T exchangeGet(String path, TypeReference<T> responseType) {
        return exchangeGet(path, body -> objectMapper.readValue(body, responseType));
    }

    private <T> T exchange(String path, Object payload, ResponseParser<T> parser) {
        ApiException lastTransportError = null;
        for (String candidateBaseUrl : candidateBaseUrls()) {
            try {
                String body = sendJson(candidateBaseUrl, path, payload);
                return parser.parse(body);
            } catch (ApiException ex) {
                if (ex.getStatus() == HttpStatus.BAD_GATEWAY) {
                    lastTransportError = ex;
                    continue;
                }
                throw ex;
            } catch (Exception ex) {
                log.error("Python AI service call failed for path '{}' at {}", path, endpoint(candidateBaseUrl, path), ex);
                throw new ApiException(HttpStatus.BAD_GATEWAY, "Failed to call Python AI service.");
            }
        }
        if (lastTransportError != null) {
            throw lastTransportError;
        }
        throw new ApiException(HttpStatus.BAD_GATEWAY, "Failed to call Python AI service.");
    }

    private <T> T exchangeGet(String path, ResponseParser<T> parser) {
        ApiException lastTransportError = null;
        for (String candidateBaseUrl : candidateBaseUrls()) {
            try {
                String body = sendGet(candidateBaseUrl, path);
                return parser.parse(body);
            } catch (ApiException ex) {
                if (ex.getStatus() == HttpStatus.BAD_GATEWAY) {
                    lastTransportError = ex;
                    continue;
                }
                throw ex;
            } catch (Exception ex) {
                log.error("Python AI service GET failed for path '{}' at {}", path, endpoint(candidateBaseUrl, path), ex);
                throw new ApiException(HttpStatus.BAD_GATEWAY, "Failed to call Python AI service.");
            }
        }
        if (lastTransportError != null) {
            throw lastTransportError;
        }
        throw new ApiException(HttpStatus.BAD_GATEWAY, "Failed to call Python AI service.");
    }

    private String sendJson(String candidateBaseUrl, String path, Object payload) {
        String url = endpoint(candidateBaseUrl, path);
        try {
            String json = objectMapper.writeValueAsString(payload);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(readTimeout)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            int statusCode = response.statusCode();
            if (statusCode >= 200 && statusCode < 300) {
                return response.body();
            }
            String detail = response.body();
            if (detail == null || detail.isBlank()) {
                detail = "HTTP " + statusCode;
            }
            log.error("Python AI service returned {} for {}: {}", statusCode, path, detail);
            HttpStatus mappedStatus = mapPythonStatus(statusCode);
            throw new ApiException(mappedStatus, "Python AI service error: " + detail);
        } catch (ApiException ex) {
            throw ex;
        } catch (ConnectException ex) {
            throw handleTransportError(path, candidateBaseUrl, ex);
        } catch (java.net.http.HttpTimeoutException ex) {
            throw handleTransportError(path, candidateBaseUrl, ex);
        } catch (java.io.IOException ex) {
            throw handleTransportError(path, candidateBaseUrl, ex);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Python AI service call was interrupted.");
        } catch (Exception ex) {
            log.error("Python AI service returned an unexpected error for {} at {}", path, url, ex);
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Failed to call Python AI service.");
        }
    }

    private String sendGet(String candidateBaseUrl, String path) {
        String url = endpoint(candidateBaseUrl, path);
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(readTimeout)
                    .header("Accept", "application/json")
                    .GET()
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            int statusCode = response.statusCode();
            if (statusCode >= 200 && statusCode < 300) {
                return response.body();
            }
            String detail = response.body();
            if (detail == null || detail.isBlank()) {
                detail = "HTTP " + statusCode;
            }
            log.error("Python AI service returned {} for {}: {}", statusCode, path, detail);
            HttpStatus mappedStatus = mapPythonStatus(statusCode);
            throw new ApiException(mappedStatus, "Python AI service error: " + detail);
        } catch (ApiException ex) {
            throw ex;
        } catch (ConnectException ex) {
            throw handleTransportError(path, candidateBaseUrl, ex);
        } catch (java.net.http.HttpTimeoutException ex) {
            throw handleTransportError(path, candidateBaseUrl, ex);
        } catch (java.io.IOException ex) {
            throw handleTransportError(path, candidateBaseUrl, ex);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Python AI service call was interrupted.");
        } catch (Exception ex) {
            log.error("Python AI service returned an unexpected error for {} at {}", path, url, ex);
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Failed to call Python AI service.");
        }
    }

    private String normalizeBaseUrl(String configuredBaseUrl) {
        if (!StringUtils.hasText(configuredBaseUrl)) {
            throw new IllegalStateException("AI service base URL is not configured.");
        }
        return configuredBaseUrl.replaceAll("/+$", "");
    }

    private List<String> candidateBaseUrls() {
        List<String> candidates = new ArrayList<>();
        candidates.add(baseUrl);
        if (baseUrl.contains("://localhost")) {
            candidates.add(baseUrl.replace("://localhost", "://127.0.0.1"));
        }
        return candidates.stream().distinct().toList();
    }

    private ApiException handleTransportError(String path, String candidateBaseUrl, Exception ex) {
        log.warn("Python AI service is unavailable for {} at {}", path, endpoint(candidateBaseUrl, path), ex);
        if (ex instanceof java.net.http.HttpTimeoutException) {
            return new ApiException(HttpStatus.GATEWAY_TIMEOUT, "Python AI service timed out.");
        }
        return new ApiException(HttpStatus.BAD_GATEWAY, "Python AI service is unavailable.");
    }

    private HttpStatus mapPythonStatus(int statusCode) {
        return switch (statusCode) {
            case 429 -> HttpStatus.TOO_MANY_REQUESTS;
            case 504 -> HttpStatus.GATEWAY_TIMEOUT;
            case 503 -> HttpStatus.SERVICE_UNAVAILABLE;
            default -> HttpStatus.BAD_GATEWAY;
        };
    }

    @FunctionalInterface
    private interface ResponseParser<T> {
        T parse(String body) throws Exception;
    }

    public record AiChatResult(
            String response,
            @JsonProperty("response_id") String responseId,
            @JsonProperty("response_hash") String responseHash,
            @JsonProperty("source_chunk_ids") List<Integer> sourceChunkIds
    ) {
        public AiChatResult normalized() {
            return new AiChatResult(
                    response == null ? "" : response.trim(),
                    responseId == null || responseId.isBlank() ? "resp_unknown" : responseId,
                    responseHash == null || responseHash.isBlank() ? "resp_unknown" : responseHash,
                    sourceChunkIds == null ? List.of() : List.copyOf(new ArrayList<>(sourceChunkIds))
            );
        }
    }

    public record SearchResponse(List<SearchMatch> matches, Integer count) {
    }

    public record SearchMatch(
            Long chunk_id,
            Long document_id,
            String file_name,
            Integer chunk_index,
            String content,
            Integer token_count,
            Double score
    ) {
    }
}
