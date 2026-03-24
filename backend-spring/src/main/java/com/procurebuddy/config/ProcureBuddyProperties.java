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
}
