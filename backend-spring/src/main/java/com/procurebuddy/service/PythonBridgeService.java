package com.procurebuddy.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.procurebuddy.config.ProcureBuddyProperties;
import com.procurebuddy.dto.response.ChatMessageResponse;
import com.procurebuddy.exception.ApiException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class PythonBridgeService {

    private final ProcureBuddyProperties properties;
    private final ObjectMapper objectMapper;

    public String askQuestion(String message, List<ChatMessageResponse> history) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("message", message);
        payload.put("history", history.stream()
                .map(item -> Map.of("role", item.role(), "content", item.content()))
                .toList());

        Map<String, Object> result = run("ask", payload);
        Object reply = result.get("reply");
        if (reply == null) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Python bridge returned an empty reply.");
        }
        return reply.toString();
    }

    public Map<String, Object> reindexKnowledgeBase() {
        return run("reindex", Map.of());
    }

    private Map<String, Object> run(String command, Map<String, Object> payload) {
        try {
            Path repoRoot = resolveRepoRoot();
            ProcessBuilder builder = new ProcessBuilder(
                    resolvePythonExecutable(repoRoot),
                    resolveBridgeScript(repoRoot).toString(),
                    command
            );
            builder.directory(repoRoot.toFile());
            configurePythonEnvironment(builder, repoRoot);

            Process process = builder.start();
            CompletableFuture<String> stdoutFuture = readStreamAsync(process.getInputStream());
            CompletableFuture<String> stderrFuture = readStreamAsync(process.getErrorStream());
            try (OutputStream output = process.getOutputStream()) {
                output.write(objectMapper.writeValueAsBytes(payload));
                output.flush();
            }

            boolean finished = process.waitFor(properties.getAiTimeoutSeconds(), TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new ApiException(HttpStatus.REQUEST_TIMEOUT, "AI response timed out.");
            }

            String rawOutput = getStreamOutput(stdoutFuture, "stdout");
            String rawError = getStreamOutput(stderrFuture, "stderr");
            int exitCode = process.exitValue();
            if (rawOutput.isBlank()) {
                String detail = rawError.isBlank() ? "Python bridge returned no output." : rawError;
                throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, detail);
            }

            Map<String, Object> result;
            try {
                result = objectMapper.readValue(rawOutput, new TypeReference<>() {
                });
            } catch (IOException ex) {
                log.error("Python bridge returned invalid JSON. stdout={}, stderr={}", rawOutput, rawError);
                String detail = rawError.isBlank() ? rawOutput : rawError;
                throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Python bridge returned invalid output: " + detail);
            }

            if (!rawError.isBlank()) {
                log.warn("Python bridge stderr for command '{}': {}", command, rawError);
            }
            if (exitCode != 0 || result.containsKey("error")) {
                String message = String.valueOf(result.getOrDefault("error", rawError.isBlank() ? rawOutput : rawError));
                throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, message);
            }
            return result;
        } catch (IOException ex) {
            log.error("Failed to execute Python bridge process", ex);
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to execute Python bridge.");
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Python bridge execution was interrupted.");
        }
    }

    private Path resolveRepoRoot() {
        return Path.of(properties.getRepoRoot()).toAbsolutePath().normalize();
    }

    private Path resolveBridgeScript(Path repoRoot) {
        Path script = repoRoot.resolve("backend-spring").resolve(properties.getPythonBridgeScript());
        if (!Files.exists(script)) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Python bridge script is missing.");
        }
        return script;
    }

    private String resolvePythonExecutable(Path repoRoot) {
        if (properties.getPythonExecutable() != null && !properties.getPythonExecutable().isBlank()) {
            return properties.getPythonExecutable();
        }

        Path windowsVenv = repoRoot.resolve("venv").resolve("Scripts").resolve("python.exe");
        if (Files.exists(windowsVenv)) {
            return windowsVenv.toString();
        }

        Path unixVenv = repoRoot.resolve("venv").resolve("bin").resolve("python");
        if (Files.exists(unixVenv)) {
            return unixVenv.toString();
        }

        return "python";
    }

    private void configurePythonEnvironment(ProcessBuilder builder, Path repoRoot) throws IOException {
        Map<String, String> environment = builder.environment();
        Path cacheRoot = repoRoot.resolve(".cache");
        Path hfHome = cacheRoot.resolve("huggingface");
        Path transformersCache = cacheRoot.resolve("transformers");
        Path sentenceTransformersHome = cacheRoot.resolve("sentence-transformers");
        Files.createDirectories(hfHome);
        Files.createDirectories(transformersCache);
        Files.createDirectories(sentenceTransformersHome);

        environment.putIfAbsent("HF_HOME", hfHome.toString());
        environment.putIfAbsent("HUGGINGFACE_HUB_CACHE", hfHome.resolve("hub").toString());
        environment.putIfAbsent("TRANSFORMERS_CACHE", transformersCache.toString());
        environment.putIfAbsent("SENTENCE_TRANSFORMERS_HOME", sentenceTransformersHome.toString());
        environment.putIfAbsent("PYTHONIOENCODING", StandardCharsets.UTF_8.name());
    }

    private CompletableFuture<String> readStreamAsync(InputStream stream) {
        return CompletableFuture.supplyAsync(() -> {
            try (InputStream input = stream) {
                return new String(input.readAllBytes(), StandardCharsets.UTF_8).trim();
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        });
    }

    private String getStreamOutput(CompletableFuture<String> future, String label) {
        try {
            return future.get();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Python bridge " + label + " read was interrupted.");
        } catch (ExecutionException ex) {
            throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to read Python bridge " + label + ".");
        }
    }
}
