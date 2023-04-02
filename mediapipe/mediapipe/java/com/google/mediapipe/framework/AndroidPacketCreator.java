// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.framework;

import android.graphics.Bitmap;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.Image;
import com.google.mediapipe.framework.image.ImageProperties;
import java.nio.ByteBuffer;

// TODO: use Preconditions in this file.
/**
 * Android-specific subclass of PacketCreator.
 *
 * <p>See {@link PacketCreator} for general information.
 *
 * <p>This class contains methods that are Android-specific. You can (and should) use the base
 * PacketCreator on Android if you do not need any methods from this class.
 */
public class AndroidPacketCreator extends PacketCreator {
  public AndroidPacketCreator(Graph context) {
    super(context);
  }

  /** Creates a 3 channel RGB ImageFrame packet from a {@link Bitmap}. */
  public Packet createRgbImageFrame(Bitmap bitmap) {
    if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
      throw new RuntimeException("bitmap must use ARGB_8888 config.");
    }
    return Packet.create(nativeCreateRgbImageFrame(mediapipeGraph.getNativeHandle(), bitmap));
  }

  /** Creates a 4 channel RGBA ImageFrame packet from a {@link Bitmap}. */
  public Packet createRgbaImageFrame(Bitmap bitmap) {
    if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
      throw new RuntimeException("bitmap must use ARGB_8888 config.");
    }
    return Packet.create(nativeCreateRgbaImageFrame(mediapipeGraph.getNativeHandle(), bitmap));
  }

  /** Creates a 4 channel RGBA Image packet from a {@link Bitmap}. */
  public Packet createRgbaImage(Bitmap bitmap) {
    if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
      throw new RuntimeException("bitmap must use ARGB_8888 config.");
    }
    return Packet.create(nativeCreateRgbaImage(mediapipeGraph.getNativeHandle(), bitmap));
  }

  /**
   * Creates an Image packet from an {@link Image}.
   *
   * <p>The ImageContainerType must be IMAGE_CONTAINER_BYTEBUFFER or IMAGE_CONTAINER_BITMAP.
   */
  public Packet createImage(Image image) {
    // TODO: Choose the best storage from multiple containers.
    ImageProperties properties = image.getContainedImageProperties().get(0);
    if (properties.getStorageType() == Image.STORAGE_TYPE_BYTEBUFFER) {
      ByteBuffer buffer = ByteBufferExtractor.extract(image);
      int numChannels = 0;
      switch (properties.getImageFormat()) {
        case Image.IMAGE_FORMAT_RGBA:
          numChannels = 4;
          break;
        case Image.IMAGE_FORMAT_RGB:
          numChannels = 3;
          break;
        case Image.IMAGE_FORMAT_ALPHA:
          numChannels = 1;
          break;
        default: // fall out
      }
      if (numChannels == 0) {
        throw new UnsupportedOperationException(
            "Unsupported MediaPipe Image image format: " + properties.getImageFormat());
      }
      int width = image.getWidth();
      int height = image.getHeight();
      return createImage(buffer, width, height, numChannels);
    }
    if (properties.getImageFormat() == Image.STORAGE_TYPE_BITMAP) {
      Bitmap bitmap = BitmapExtractor.extract(image);
      if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
        throw new UnsupportedOperationException("bitmap must use ARGB_8888 config.");
      }
      return Packet.create(nativeCreateRgbaImage(mediapipeGraph.getNativeHandle(), bitmap));
    }

    // Unsupported type.
    throw new UnsupportedOperationException(
        "Unsupported Image container type: " + properties.getImageFormat());
  }

  /**
   * Returns the native handle of a new internal::PacketWithContext object on success. Returns 0 on
   * failure.
   */
  private native long nativeCreateRgbImageFrame(long context, Bitmap bitmap);

  /**
   * Returns the native handle of a new internal::PacketWithContext object on success. Returns 0 on
   * failure.
   */
  private native long nativeCreateRgbaImageFrame(long context, Bitmap bitmap);

  /**
   * Returns the native handle of a new internal::PacketWithContext object on success. Returns 0 on
   * failure.
   */
  private native long nativeCreateRgbaImage(long context, Bitmap bitmap);
}
