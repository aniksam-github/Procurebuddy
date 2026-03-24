package com.procurebuddy.util;

public final class PasswordRules {

    public static final String REQUIREMENT_MESSAGE =
            "Password must be at least 8 characters and include uppercase, lowercase, number, and special symbol.";

    private PasswordRules() {
    }

    public static boolean isStrong(String password) {
        if (password == null || password.length() < 8) {
            return false;
        }
        boolean uppercase = false;
        boolean lowercase = false;
        boolean digit = false;
        boolean symbol = false;

        for (char value : password.toCharArray()) {
            if (Character.isUpperCase(value)) {
                uppercase = true;
            } else if (Character.isLowerCase(value)) {
                lowercase = true;
            } else if (Character.isDigit(value)) {
                digit = true;
            } else {
                symbol = true;
            }
        }

        return uppercase && lowercase && digit && symbol;
    }
}
