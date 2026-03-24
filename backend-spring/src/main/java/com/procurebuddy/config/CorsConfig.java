package com.procurebuddy.config;

import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
@RequiredArgsConstructor
public class CorsConfig implements WebMvcConfigurer {

    private final ProcureBuddyProperties properties;

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        List<String> origins = properties.getCors().getAllowedOrigins();
        List<String> originPatterns = properties.getCors().getAllowedOriginPatterns();

        var registration = registry.addMapping("/**")
                .allowedMethods("*")
                .allowedHeaders("*");

        if (!origins.isEmpty()) {
            registration.allowedOrigins(origins.toArray(String[]::new));
        }

        if (!originPatterns.isEmpty()) {
            registration.allowedOriginPatterns(originPatterns.toArray(String[]::new));
        }
    }
}
