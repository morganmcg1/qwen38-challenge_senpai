#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

int main(void) {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) {
      printf("NO_METAL_DEVICE\n");
      return 1;
    }
    NSString *name = [d name];
    printf("device.name         = %s\n", [name UTF8String]);
    if (@available(macOS 14.0, *)) {
      NSString *arch = [[d architecture] name];
      const char *a = [arch UTF8String];
      size_t n = strlen(a);
      printf("architecture.name   = %s\n", a);
      if (n >= 3) {
        int tens = a[n - 3] - '0';
        int ones = a[n - 2] - '0';
        printf("MLX arch_gen        = %d\n", tens * 10 + ones);
        printf("MLX devc (back)     = '%c'\n", a[n - 1]);
        int two_pass = (a[n - 1] == 'd' || a[n - 1] == 's');
        printf("SDPA 2pass at KV>=1024 on this box? %s\n",
               two_pass ? "YES" : "NO");
      }
    } else {
      printf("architecture unavailable on this OS\n");
    }
  }
  return 0;
}
