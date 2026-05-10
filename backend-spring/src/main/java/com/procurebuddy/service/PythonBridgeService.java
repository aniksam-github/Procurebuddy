package com.procurebuddy.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.procurebuddy.config.ProcureBuddyProperties;
import com.procurebuddy.dto.response.ChatMessageResponse;
import com.procurebuddy.exception.ApiException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Slf4j
@Service
public class PythonBridgeService {

    private static final String RATE_LIMIT_USER_MESSAGE =
            "Groq API rate limit hit ho gayi hai. Please thodi der baad phir try karein.";

    private final ProcureBuddyProperties properties;
    private final ObjectMapper objectMapper;

    public PythonBridgeService(ProcureBuddyProperties properties, ObjectMapper objectMapper) {
        this.properties = properties;
        this.objectMapper = objectMapper;
    }

    public String askQuestion(String message, List<ChatMessageResponse> history) {
        String baseUrl = properties.getPythonService().getBaseUrl();
        int readTimeoutSeconds = properties.getPythonService().getReadTimeoutSeconds();

        Map<String, Object> payload = new HashMap<>();
        payload.put("message", message);
        payload.put("user", "spring-bridge");

        try {
            String requestBody = objectMapper.writeValueAsString(payload);
            log.debug("Sending payload to AI service: {}", requestBody);

            log.info("Sending chat request to Python AI service: {}/chat (timeout={}s)", baseUrl, readTimeoutSeconds);
            long start = System.currentTimeMillis();

            HttpResponseData response = executeJsonPost(baseUrl + "/chat", requestBody, readTimeoutSeconds);

            long elapsed = System.currentTimeMillis() - start;
            log.info("Python AI service responded in {}ms with status {}", elapsed, response.statusCode());

            if (response.statusCode() == 429) {
                log.warn("Python AI service returned 429 rate limit");
                throw new ApiException(HttpStatus.TOO_MANY_REQUESTS, RATE_LIMIT_USER_MESSAGE);
            }

            if (response.statusCode() >= 500) {
                log.error("Python AI service returned server error {}: {}", response.statusCode(), response.body());
                throw new ApiException(HttpStatus.BAD_GATEWAY, "AI service error: " + response.statusCode());
            }

            if (response.statusCode() != 200) {
                log.error("Python AI service returned unexpected status {}: {}", response.statusCode(), response.body());
                throw new ApiException(HttpStatus.BAD_GATEWAY, "AI service returned status " + response.statusCode());
            }

            Map<String, Object> result = objectMapper.readValue(response.body(), new TypeReference<>() {});

            String answer = String.valueOf(result.getOrDefault("answer", result.getOrDefault("response", "")));
            if (answer.isBlank() || "null".equals(answer)) {
                log.warn("Python AI service returned empty answer");
                throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "AI service returned an empty reply.");
            }

            return answer;

        } catch (ApiException ex) {
            throw ex;
        } catch (SocketTimeoutException ex) {
            log.error("Python AI service timed out after {}s", readTimeoutSeconds, ex);
            throw new ApiException(HttpStatus.REQUEST_TIMEOUT, "AI response timed out.");
        } catch (IOException | InterruptedException ex) {
            if (ex instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            log.error("Failed to connect to Python AI service at {}", baseUrl, ex);
            throw new ApiException(
                HttpStatus.BAD_GATEWAY,
                "Could not reach AI service. Make sure the Python backend is running on " + baseUrl
            );
        }
    }

    public Map<String, Object> reindexKnowledgeBase() {
        String baseUrl = properties.getPythonService().getBaseUrl();
        try {
            HttpResponseData response = executeJsonPost(baseUrl + "/reload", "{}", 300);

            if (response.statusCode() != 200) {
                throw new ApiException(HttpStatus.BAD_GATEWAY, "Reindex failed with status " + response.statusCode());
            }

            return objectMapper.readValue(response.body(), new TypeReference<>() {});

        } catch (ApiException ex) {
            throw ex;
        } catch (IOException | InterruptedException ex) {
            if (ex instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            log.error("Failed to call /reload on Python AI service", ex);
            throw new ApiException(HttpStatus.BAD_GATEWAY, "Could not reach AI service for reindex.");
        }
    }

    private HttpResponseData executeJsonPost(String url, String requestBody, int timeoutSeconds)
            throws IOException, InterruptedException {
        HttpURLConnection connection = openConnection(url, "POST", timeoutSeconds);
        connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");

        try {
            byte[] bodyBytes = requestBody.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            connection.setFixedLengthStreamingMode(bodyBytes.length);
            try (OutputStream outputStream = connection.getOutputStream()) {
                outputStream.write(bodyBytes);
            }

            int statusCode = connection.getResponseCode();
            String responseBody = readResponseBody(connection, statusCode);
            return new HttpResponseData(statusCode, responseBody);
        } finally {
            connection.disconnect();
        }
    }

    private HttpURLConnection openConnection(String url, String method, int timeoutSeconds) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) java.net.URI.create(url).toURL().openConnection();
        int timeoutMillis = Math.toIntExact(Duration.ofSeconds(timeoutSeconds).toMillis());
        connection.setRequestMethod(method);
        connection.setConnectTimeout(timeoutMillis);
        connection.setReadTimeout(timeoutMillis);
        connection.setRequestProperty("Accept", "application/json");
        return connection;
    }

    private String readResponseBody(HttpURLConnection connection, int statusCode) throws IOException {
        InputStream stream = statusCode >= 400 ? connection.getErrorStream() : connection.getInputStream();
        if (stream == null) {
            return "";
        }
        try (InputStream inputStream = stream) {
            return new String(inputStream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    private record HttpResponseData(int statusCode, String body) {
    }
}
