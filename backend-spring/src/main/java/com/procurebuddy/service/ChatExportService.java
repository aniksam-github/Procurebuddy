package com.procurebuddy.service;

import com.procurebuddy.entity.ChatEntity;
import com.procurebuddy.entity.MessageEntity;
import com.procurebuddy.entity.UserEntity;
import com.procurebuddy.exception.ApiException;
import com.procurebuddy.repository.ChatRepository;
import com.procurebuddy.repository.MessageRepository;
import com.procurebuddy.util.UserResolver;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class ChatExportService {

    private static final float PAGE_WIDTH = 612f;
    private static final float PAGE_HEIGHT = 792f;
    private static final float PAGE_MARGIN = 56f;
    private static final float TITLE_FONT_SIZE = 18f;
    private static final float LABEL_FONT_SIZE = 10f;
    private static final float BODY_FONT_SIZE = 11f;
    private static final float LINE_GAP = 4f;
    private static final DateTimeFormatter TIMESTAMP_FORMATTER = DateTimeFormatter.ofPattern("dd MMM yyyy, hh:mm a");

    private final ChatRepository chatRepository;
    private final MessageRepository messageRepository;
    private final UserResolver userResolver;

    @Transactional(readOnly = true)
    public ChatExportResult exportChatPdf(String chatId, String email) {
        UserEntity user = userResolver.requireByEmail(email);
        ChatEntity chat = chatRepository.findByIdAndUser(chatId, user)
                .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "Chat not found."));
        List<MessageEntity> exchanges = messageRepository.findAllByChatOrderByTimestampAscIdAsc(chat);

        PdfDocumentBuilder builder = new PdfDocumentBuilder();
        builder.writeHeading((chat.getTitle() == null || chat.getTitle().isBlank()) ? "ProcureBuddy Chat Export" : chat.getTitle());

        if (exchanges.isEmpty()) {
            builder.writeParagraph("This chat does not have any messages yet.", BODY_FONT_SIZE, false, 18f);
        }

        for (MessageEntity exchange : exchanges) {
            builder.writeLabel("User", exchange.getTimestamp() == null ? null : TIMESTAMP_FORMATTER.format(exchange.getTimestamp()));
            builder.writeParagraph(exchange.getMessage(), BODY_FONT_SIZE, false, 14f);
            builder.writeLabel("ProcureBuddy", exchange.getTimestamp() == null ? null : TIMESTAMP_FORMATTER.format(exchange.getTimestamp()));
            builder.writeParagraph(exchange.getResponse(), BODY_FONT_SIZE, false, 22f);
        }

        String filename = sanitizeFilename((chat.getTitle() == null || chat.getTitle().isBlank()) ? chat.getId() : chat.getTitle()) + ".pdf";
        return new ChatExportResult(filename, builder.build());
    }

    private String sanitizeFilename(String value) {
        String normalized = value == null ? "procurebuddy-chat" : value.replaceAll("[^a-zA-Z0-9_ -]", "").trim();
        if (normalized.isBlank()) {
            return "procurebuddy-chat";
        }
        return normalized.replaceAll("\\s+", "-").toLowerCase();
    }

    public record ChatExportResult(String filename, byte[] content) {
    }

    private static final class PdfDocumentBuilder {
        private final List<StringBuilder> pages = new ArrayList<>();
        private StringBuilder currentPage;
        private float y;

        private PdfDocumentBuilder() {
            newPage();
        }

        private void writeHeading(String text) {
            ensureSpace(32f);
            appendText("F2", TITLE_FONT_SIZE, PAGE_MARGIN, y, text);
            y -= 28f;
        }

        private void writeLabel(String role, String timestamp) {
            ensureSpace(18f);
            appendText("F2", LABEL_FONT_SIZE, PAGE_MARGIN, y, timestamp == null ? role : role + " - " + timestamp);
            y -= 14f;
        }

        private void writeParagraph(String text, float fontSize, boolean bold, float marginAfter) {
            List<String> lines = wrapText(text, fontSize);
            for (String line : lines) {
                ensureSpace(fontSize + 6f);
                appendText(bold ? "F2" : "F1", fontSize, PAGE_MARGIN, y, line);
                y -= fontSize + LINE_GAP;
            }
            y -= marginAfter;
        }

        private void ensureSpace(float requiredHeight) {
            if (y - requiredHeight <= PAGE_MARGIN) {
                newPage();
            }
        }

        private void newPage() {
            currentPage = new StringBuilder();
            pages.add(currentPage);
            y = PAGE_HEIGHT - PAGE_MARGIN;
        }

        private List<String> wrapText(String text, float fontSize) {
            String normalized = (text == null || text.isBlank()) ? "-" : text.strip().replace("\r", "");
            List<String> lines = new ArrayList<>();
            int maxChars = Math.max(18, (int) Math.floor((PAGE_WIDTH - (PAGE_MARGIN * 2f)) / Math.max(5.2, fontSize * 0.52)));
            for (String paragraph : normalized.split("\n")) {
                String trimmed = paragraph.isBlank() ? "-" : paragraph.strip();
                StringBuilder line = new StringBuilder();
                for (String word : trimmed.split("\\s+")) {
                    String candidate = line.isEmpty() ? word : line + " " + word;
                    if (candidate.length() > maxChars && !line.isEmpty()) {
                        lines.add(line.toString());
                        line = new StringBuilder(word);
                    } else {
                        line = new StringBuilder(candidate);
                    }
                }
                if (!line.isEmpty()) {
                    lines.add(line.toString());
                }
            }
            return lines;
        }

        private void appendText(String fontKey, float fontSize, float x, float yPosition, String text) {
            currentPage.append("BT\n")
                    .append("/")
                    .append(fontKey)
                    .append(" ")
                    .append(format(fontSize))
                    .append(" Tf\n")
                    .append(format(x))
                    .append(" ")
                    .append(format(yPosition))
                    .append(" Td\n")
                    .append("(")
                    .append(escape(text))
                    .append(") Tj\n")
                    .append("ET\n");
        }

        private byte[] build() {
            try {
                List<String> objects = new ArrayList<>();
                objects.add("<< /Type /Catalog /Pages 2 0 R >>");

                StringBuilder kids = new StringBuilder();
                for (int index = 0; index < pages.size(); index++) {
                    int pageObjectNumber = 3 + (index * 2);
                    kids.append(pageObjectNumber).append(" 0 R ");
                }
                objects.add("<< /Type /Pages /Kids [" + kids + "] /Count " + pages.size() + " >>");

                int fontObjectNumber = 3 + (pages.size() * 2);
                int boldFontObjectNumber = fontObjectNumber + 1;
                for (int index = 0; index < pages.size(); index++) {
                    int pageObjectNumber = 3 + (index * 2);
                    int contentObjectNumber = pageObjectNumber + 1;
                    objects.add(
                            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + format(PAGE_WIDTH) + " " + format(PAGE_HEIGHT) + "] "
                                    + "/Resources << /Font << /F1 " + fontObjectNumber + " 0 R /F2 " + boldFontObjectNumber + " 0 R >> >> "
                                    + "/Contents " + contentObjectNumber + " 0 R >>"
                    );

                    byte[] streamBytes = pages.get(index).toString().getBytes(StandardCharsets.US_ASCII);
                    objects.add("<< /Length " + streamBytes.length + " >>\nstream\n" + pages.get(index) + "endstream");
                }

                objects.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
                objects.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>");

                ByteArrayOutputStream output = new ByteArrayOutputStream();
                output.write("%PDF-1.4\n".getBytes(StandardCharsets.US_ASCII));
                List<Integer> offsets = new ArrayList<>();
                offsets.add(0);

                for (int index = 0; index < objects.size(); index++) {
                    offsets.add(output.size());
                    String object = (index + 1) + " 0 obj\n" + objects.get(index) + "\nendobj\n";
                    output.write(object.getBytes(StandardCharsets.US_ASCII));
                }

                int xrefOffset = output.size();
                output.write(("xref\n0 " + (objects.size() + 1) + "\n").getBytes(StandardCharsets.US_ASCII));
                output.write("0000000000 65535 f \n".getBytes(StandardCharsets.US_ASCII));
                for (int index = 1; index < offsets.size(); index++) {
                    output.write(String.format("%010d 00000 n \n", offsets.get(index)).getBytes(StandardCharsets.US_ASCII));
                }
                output.write(("trailer\n<< /Size " + (objects.size() + 1) + " /Root 1 0 R >>\n").getBytes(StandardCharsets.US_ASCII));
                output.write(("startxref\n" + xrefOffset + "\n%%EOF").getBytes(StandardCharsets.US_ASCII));
                return output.toByteArray();
            } catch (Exception ex) {
                throw new ApiException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to export chat.");
            }
        }

        private String escape(String value) {
            String sanitized = (value == null || value.isBlank()) ? "-" : value;
            sanitized = sanitized.replace("\\", "\\\\")
                    .replace("(", "\\(")
                    .replace(")", "\\)");
            StringBuilder ascii = new StringBuilder(sanitized.length());
            for (char character : sanitized.toCharArray()) {
                ascii.append(character <= 0x7F ? character : '?');
            }
            return ascii.toString();
        }

        private String format(float value) {
            return String.format(java.util.Locale.US, "%.2f", value);
        }
    }
}
