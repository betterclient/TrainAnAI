package io.github.betterclient.ai;

import io.github.betterclient.ai.model.digit.DigitRecognitionModel;
import org.json.JSONObject;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Parse MNIST database for {@link DigitRecognitionModel}
 */
public class MNISTParser {
    public static void main(String[] args) throws Exception {
        File f = new File("D:\\DOWNLOADS\\MNIST Dataset\\MNIST Dataset\\MNIST - JPG - training");
        File out = new File("D:\\DOWNLOADS\\MNIST Dataset\\parsed");

        for (File file : Objects.requireNonNull(f.listFiles())) {
            List<JSONObject> outt = new ArrayList<>();
            int index = 0;
            for (File listFile : Objects.requireNonNull(file.listFiles())) {
                if (index > 100) {
                    break;
                }

                BufferedImage image = resize(ImageIO.read(listFile));

                JSONObject object = new JSONObject();
                for (int x = 0; x < 16; x++) {
                    for (int y = 0; y < 16; y++) {
                        int rgb = image.getRGB(x, y);
                        int red = (rgb >> 16) & 0xFF;
                        int green = (rgb >> 8) & 0xFF;
                        int blue = rgb & 0xFF;
                        int brightness = (int) (0.299 * red + 0.587 * green + 0.114 * blue);

                        object.put(x + "," + y, brightness > 128);
                    }
                }
                outt.add(object);

                index++;
            }

            FileOutputStream fos = new FileOutputStream(new File(out, file.getName() + ".json"));
            fos.write(outt.stream().map(JSONObject::toString).collect(Collectors.joining("/")).getBytes());
            fos.close();
        }
    }

    private static BufferedImage resize(BufferedImage originalImage) {
        BufferedImage resizedImage = new BufferedImage(16, 16, originalImage.getType());

        Graphics2D g2d = resizedImage.createGraphics();

        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2d.drawImage(originalImage, 0, 0, 16, 16, null);

        g2d.dispose();

        return resizedImage;
    }
}
