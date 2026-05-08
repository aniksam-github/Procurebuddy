package com.procurebuddy.persistence;

import jakarta.persistence.AttributeConverter;
import jakarta.persistence.Converter;
import java.util.Locale;

@Converter
public class FloatArrayStringConverter implements AttributeConverter<float[], String> {

    @Override
    public String convertToDatabaseColumn(float[] attribute) {
        if (attribute == null || attribute.length == 0) {
            return "";
        }

        StringBuilder serialized = new StringBuilder(attribute.length * 8);
        for (int index = 0; index < attribute.length; index++) {
            if (index > 0) {
                serialized.append(',');
            }
            serialized.append(String.format(Locale.ROOT, "%.7f", attribute[index]));
        }
        return serialized.toString();
    }

    @Override
    public float[] convertToEntityAttribute(String dbData) {
        if (dbData == null || dbData.isBlank()) {
            return new float[0];
        }

        String[] parts = dbData.split(",");
        float[] values = new float[parts.length];
        for (int index = 0; index < parts.length; index++) {
            values[index] = Float.parseFloat(parts[index]);
        }
        return values;
    }
}
