package com.procurebuddy.config;

import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
@ConfigurationProperties(prefix = "procurebuddy")
public class ProcureBuddyProperties {

    private Cors cors = new Cors();
    private String adminEmail;
    private String dataDir;
    private PythonService pythonService = new PythonService();
    private Jwt jwt = new Jwt();
    private String repoRoot;
    private String pythonExecutable;
    private String pythonBridgeScript;
    private int aiTimeoutSeconds = 90;
    private Async async = new Async();

    @Getter
    @Setter
    public static class Cors {
        private List<String> allowedOrigins = new ArrayList<>();
        private List<String> allowedOriginPatterns = new ArrayList<>();
    }

    @Getter
    @Setter
    public static class Async {
        private int corePoolSize = 16;
        private int maxPoolSize = 32;
        private int queueCapacity = 400;
        private String threadNamePrefix = "procurebuddy-async-";
    }

    @Getter
    @Setter
    public static class PythonService {
        private String baseUrl = "http://127.0.0.1:8000";
        private int connectTimeoutSeconds = 10;
        private int readTimeoutSeconds = 180;
    }

    @Getter
    @Setter
    public static class Jwt {
        private String secret;
        private int accessTokenMinutes = 720;
        private int challengeTokenMinutes = 10;
    }
}
