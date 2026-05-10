package com.procurebuddy.security;

import com.procurebuddy.config.ProcureBuddyProperties;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;
import javax.crypto.SecretKey;
import org.springframework.stereotype.Service;

@Service
public class JwtService {

    private static final String CLAIM_TYPE = "type";
    private static final String CLAIM_ADMIN = "admin";
    private static final String CLAIM_PURPOSE = "purpose";
    private static final String TOKEN_TYPE_ACCESS = "access";
    private static final String TOKEN_TYPE_CHALLENGE = "challenge";

    private final ProcureBuddyProperties properties;
    private final SecretKey signingKey;

    public JwtService(ProcureBuddyProperties properties) {
        this.properties = properties;
        String secret = properties.getJwt().getSecret();
        if (secret == null || secret.trim().length() < 32) {
            throw new IllegalStateException("JWT_SECRET must be set to at least 32 characters.");
        }
        this.signingKey = Keys.hmacShaKeyFor(secret.trim().getBytes(StandardCharsets.UTF_8));
    }

    public String issueAccessToken(String email, boolean isAdmin) {
        Instant now = Instant.now();
        Instant expiry = now.plus(properties.getJwt().getAccessTokenMinutes(), ChronoUnit.MINUTES);
        return Jwts.builder()
                .subject(email)
                .claim(CLAIM_TYPE, TOKEN_TYPE_ACCESS)
                .claim(CLAIM_ADMIN, isAdmin)
                .issuedAt(Date.from(now))
                .expiration(Date.from(expiry))
                .signWith(signingKey)
                .compact();
    }

    public String issueChallengeToken(String email, String purpose) {
        Instant now = Instant.now();
        Instant expiry = now.plus(properties.getJwt().getChallengeTokenMinutes(), ChronoUnit.MINUTES);
        return Jwts.builder()
                .subject(email)
                .claim(CLAIM_TYPE, TOKEN_TYPE_CHALLENGE)
                .claim(CLAIM_PURPOSE, purpose)
                .issuedAt(Date.from(now))
                .expiration(Date.from(expiry))
                .signWith(signingKey)
                .compact();
    }

    public String extractAccessSubject(String token) {
        Claims claims = parse(token);
        if (!TOKEN_TYPE_ACCESS.equals(claims.get(CLAIM_TYPE, String.class))) {
            throw new JwtException("Invalid access token.");
        }
        return claims.getSubject();
    }

    public String extractChallengeSubject(String token, String purpose) {
        Claims claims = parse(token);
        if (!TOKEN_TYPE_CHALLENGE.equals(claims.get(CLAIM_TYPE, String.class))) {
            throw new JwtException("Invalid challenge token.");
        }
        if (!purpose.equals(claims.get(CLAIM_PURPOSE, String.class))) {
            throw new JwtException("Invalid challenge token purpose.");
        }
        return claims.getSubject();
    }

    public boolean isAccessToken(String token) {
        try {
            return TOKEN_TYPE_ACCESS.equals(parse(token).get(CLAIM_TYPE, String.class));
        } catch (JwtException | IllegalArgumentException ex) {
            return false;
        }
    }

    private Claims parse(String token) {
        return Jwts.parser()
                .verifyWith(signingKey)
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }
}
