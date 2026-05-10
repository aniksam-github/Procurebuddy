package com.procurebuddy.security;

import io.jsonwebtoken.JwtException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
@Slf4j
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtService jwtService;

    public JwtAuthenticationFilter(JwtService jwtService) {
        this.jwtService = jwtService;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
            throws ServletException, IOException {
        String authorization = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (authorization != null && authorization.startsWith("Bearer ") && SecurityContextHolder.getContext().getAuthentication() == null) {
            String token = authorization.substring(7).trim();
            if (!token.isEmpty()) {
                try {
                    String email = jwtService.extractAccessSubject(token);
                    var authentication = new UsernamePasswordAuthenticationToken(
                            email,
                            null,
                            List.of(new SimpleGrantedAuthority("ROLE_USER"))
                    );
                    authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                    log.debug("Authenticated JWT request: path={} subject={}", request.getRequestURI(), email);
                } catch (JwtException | IllegalArgumentException ex) {
                    log.warn("JWT rejected for path {}: {}", request.getRequestURI(), ex.getMessage());
                    SecurityContextHolder.clearContext();
                }
            }
        } else if (authorization != null && request.getRequestURI().startsWith("/api/chats")) {
            log.warn("Authorization header is not a Bearer token for path {}", request.getRequestURI());
        } else if (request.getRequestURI().startsWith("/api/chats")) {
            log.warn("No bearer token found for path {}", request.getRequestURI());
        }
        filterChain.doFilter(request, response);
    }
}
