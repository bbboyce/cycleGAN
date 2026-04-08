#!/usr/bin/env python3
"""
EdUHK Rain vs Sunny Dataset Collector
香港教育大学 - 雨天vs晴天图像数据集收集工具

支持多种数据源:
1. Bing Image Search API (推荐)
2. Unsplash API
3. Pexels API
4. 手动下载
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from typing import List, Optional
import shutil

class DatasetCollector:
    def __init__(self, output_dir: str = "./datasets/edhuk_weather"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """创建CycleGAN所需的目录结构"""
        # 未配对数据集结构 (CycleGAN通常使用这个)
        dirs = [
            'trainA',  # 雨天训练图像
            'trainB',  # 晴天训练图像
            'testA',   # 雨天测试图像
            'testB',   # 晴天测试图像
        ]
        for dirname in dirs:
            (self.output_dir / dirname).mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {self.output_dir / dirname}")
    
    def download_from_bing(self, query: str, num_images: int = 250, 
                          save_dir: str = 'trainA', delay: float = 0.5):
        """
        使用Bing Image Search下载图像
        
        Args:
            query: 搜索关键词 (如 "Hong Kong Education University rain")
            num_images: 要下载的图像数量
            save_dir: 保存目录 (trainA 或 trainB)
            delay: 请求间隔 (秒)
        """
        print(f"\n📥 Bing下载: {query}")
        print(f"   目标数量: {num_images}")
        
        try:
            from bing_image_downloader import bing_image_downloader
            
            bing_downloader = bing_image_downloader.bing_image_downloader(
                query=query,
                limit=num_images,
                output_dir="dataset",
                adult_filter_off=True,
                force_replace=False,
                timeout=15,
                verbose=False
            )
            
            # 移动下载的图像到目标目录
            source_dir = Path(f"dataset/{query}")
            target_dir = self.output_dir / save_dir
            
            if source_dir.exists():
                count = 0
                for img_file in source_dir.glob("*.jpg"):
                    try:
                        shutil.copy2(str(img_file), str(target_dir / img_file.name))
                        count += 1
                    except Exception as e:
                        print(f"⚠️  复制失败: {e}")
                
                print(f"✅ 成功下载 {count} 张图像到 {save_dir}/")
                shutil.rmtree(source_dir)
            
        except ImportError:
            print("❌ 需要安装: pip install bing-image-downloader")
            self.print_manual_download_guide()
    
    def download_from_unsplash(self, query: str, num_images: int = 250, 
                               save_dir: str = 'trainA'):
        """
        使用Unsplash API下载图像 (需要API Key)
        """
        print(f"\n📥 Unsplash下载: {query}")
        print("   提示: 需要Unsplash API Key (访问 https://unsplash.com/developers)")
        
        api_key = input("请输入Unsplash API Key (或按Enter跳过): ").strip()
        if not api_key:
            print("⏭️  跳过Unsplash下载")
            return
        
        headers = {"Authorization": f"Client-ID {api_key}"}
        params = {
            "query": query,
            "per_page": min(num_images, 30),  # API限制30/页
            "page": 1,
            "order_by": "relevant"
        }
        
        try:
            response = requests.get(
                "https://api.unsplash.com/search/photos",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            count = 0
            target_dir = self.output_dir / save_dir
            
            for idx, photo in enumerate(data.get('results', [])[:num_images]):
                try:
                    img_url = photo['urls']['regular']
                    img_response = requests.get(img_url, timeout=10)
                    img_response.raise_for_status()
                    
                    filename = f"{query.replace(' ', '_')}_{idx:04d}.jpg"
                    filepath = target_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    count += 1
                    print(f"  ✅ {count}/{num_images}", end='\r')
                    
                except Exception as e:
                    print(f"  ⚠️  下载失败: {e}")
            
            print(f"\n✅ 成功下载 {count} 张图像到 {save_dir}/")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    def download_from_pexels(self, query: str, num_images: int = 250, 
                             save_dir: str = 'trainA'):
        """
        使用Pexels API下载图像 (需要API Key)
        """
        print(f"\n📥 Pexels下载: {query}")
        print("   提示: 需要Pexels API Key (访问 https://www.pexels.com/api/)")
        
        api_key = input("请输入Pexels API Key (或按Enter跳过): ").strip()
        if not api_key:
            print("⏭️  跳过Pexels下载")
            return
        
        headers = {"Authorization": api_key}
        params = {
            "query": query,
            "per_page": min(num_images, 80),
            "page": 1
        }
        
        try:
            response = requests.get(
                "https://api.pexels.com/v1/search",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            count = 0
            target_dir = self.output_dir / save_dir
            
            for idx, photo in enumerate(data.get('photos', [])[:num_images]):
                try:
                    img_url = photo['src']['large']
                    img_response = requests.get(img_url, timeout=10)
                    img_response.raise_for_status()
                    
                    filename = f"{query.replace(' ', '_')}_{idx:04d}.jpg"
                    filepath = target_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)
                    
                    count += 1
                    print(f"  ✅ {count}/{num_images}", end='\r')
                    
                except Exception as e:
                    print(f"  ⚠️  下载失败: {e}")
            
            print(f"\n✅ 成功下载 {count} 张图像到 {save_dir}/")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    def print_manual_download_guide(self):
        """打印手动下载指南"""
        guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   📥 手动下载图像指南                                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📍 推荐搜索关键词:

【雨天图像 (trainA)】
- "Hong Kong Education University rainy day"
- "Hong Kong rain weather street"
- "university campus rain"
- "Hong Kong rainy streets"
- "rain weather photography urban"

【晴天图像 (trainB)】
- "Hong Kong Education University sunny"
- "Hong Kong sunny weather street"
- "university campus sunny day"
- "Hong Kong clear sky"
- "sunny weather photography urban"

📌 推荐来源:
1. Unsplash (unsplash.com) - 免费无版权
2. Pexels (pexels.com) - 免费无版权
3. Pixabay (pixabay.com) - 免费无版权
4. Google Images - 带过滤器

✅ 下载步骤:
1. 访问上述网站
2. 搜索关键词并过滤为JPG格式
3. 批量下载图像
4. 将雨天图像放入: datasets/edhuk_weather/trainA/
5. 将晴天图像放入: datasets/edhuk_weather/trainB/
6. 分别将部分图像收入 testA/ 和 testB/ (约50-100张)

📊 目标结构:
edhuk_weather/
├── trainA/  (200-250张 雨天图像)
├── trainB/  (200-250张 晴天图像)
├── testA/   (50张 雨天图像)
└── testB/   (50张 晴天图像)

⚠️  注意事项:
- 确保图像分辨率至少 256x256 (建议 512x512)
- 图像应该是高质量的 (jpg, png, bmp等)
- 优先选择城市/校园场景
- 确保雨天和晴天图像有相似的视角和照明条件
"""
        print(guide)
    
    def validate_dataset(self) -> dict:
        """验证数据集完整性"""
        print("\n📋 验证数据集...")
        print("="*60)
        
        stats = {}
        for dirname in ['trainA', 'trainB', 'testA', 'testB']:
            dirpath = self.output_dir / dirname
            if dirpath.exists():
                files = list(dirpath.glob('*.jpg')) + list(dirpath.glob('*.png'))
                stats[dirname] = len(files)
                print(f"  {dirname:10} : {len(files):3} 张图像")
            else:
                stats[dirname] = 0
        
        total = sum(stats.values())
        print("="*60)
        print(f"  总计: {total:3} 张图像")
        
        # 检查图像质量
        print("\n📊 检查图像质量...")
        for dirname in ['trainA', 'trainB', 'testA', 'testB']:
            dirpath = self.output_dir / dirname
            if dirpath.exists():
                files = list(dirpath.glob('*.jpg')) + list(dirpath.glob('*.png'))
                if files:
                    try:
                        from PIL import Image
                        sizes = []
                        for img_file in files[:3]:  # 检查前3张
                            try:
                                img = Image.open(img_file)
                                sizes.append(img.size)
                            except:
                                pass
                        if sizes:
                            avg_size = tuple(int(sum(x)/len(x)) for x in zip(*sizes))
                            print(f"  {dirname:10} : 平均分辨率 {avg_size[0]}x{avg_size[1]}")
                    except ImportError:
                        pass
        
        return stats
    
    def create_info_json(self):
        """创建数据集信息文件"""
        info = {
            "name": "EdUHK Rain vs Sunny",
            "description": "Hong Kong Education University weather dataset",
            "source": "EdUHK Campus",
            "domain_A": "rainy",
            "domain_B": "sunny",
            "image_count": {
                "trainA": len(list((self.output_dir / 'trainA').glob('*.jpg'))),
                "trainB": len(list((self.output_dir / 'trainB').glob('*.jpg'))),
                "testA": len(list((self.output_dir / 'testA').glob('*.jpg'))),
                "testB": len(list((self.output_dir / 'testB').glob('*.jpg'))),
            }
        }
        
        info_file = self.output_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"\n✅ 数据集信息已保存: {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="EdUHK Rain vs Sunny Dataset Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Bing下载
  python prepare_edhuk_dataset.py --bing --num-images 500
  
  # Unsplash下载 (需要API Key)
  python prepare_edhuk_dataset.py --unsplash --api-key YOUR_KEY
  
  # 显示手动下载指南
  python prepare_edhuk_dataset.py --manual
  
  # 验证现有数据集
  python prepare_edhuk_dataset.py --validate
        """
    )
    
    parser.add_argument('--output-dir', default='./datasets/edhuk_weather',
                       help='输出目录 (默认: ./datasets/edhuk_weather)')
    parser.add_argument('--bing', action='store_true',
                       help='使用Bing Image Search下载')
    parser.add_argument('--unsplash', action='store_true',
                       help='使用Unsplash API下载')
    parser.add_argument('--pexels', action='store_true',
                       help='使用Pexels API下载')
    parser.add_argument('--manual', action='store_true',
                       help='显示手动下载指南')
    parser.add_argument('--validate', action='store_true',
                       help='验证现有数据集')
    parser.add_argument('--num-images', type=int, default=250,
                       help='每种天气类型的图像数 (默认: 250)')
    parser.add_argument('--api-key', type=str,
                       help='API Key (Unsplash或Pexels)')
    
    args = parser.parse_args()
    
    collector = DatasetCollector(args.output_dir)
    
    if args.validate:
        collector.validate_dataset()
        collector.create_info_json()
    elif args.manual:
        collector.print_manual_download_guide()
    elif args.bing:
        print("\n🔧 Bing Image Search 下载")
        print("="*60)
        collector.download_from_bing(
            "Hong Kong Education University rainy rain weather",
            num_images=args.num_images,
            save_dir='trainA'
        )
        collector.download_from_bing(
            "Hong Kong Education University sunny clear weather",
            num_images=args.num_images,
            save_dir='trainB'
        )
        collector.validate_dataset()
        collector.create_info_json()
    elif args.unsplash:
        collector.download_from_unsplash(
            "Hong Kong rain rainy weather urban",
            num_images=args.num_images,
            save_dir='trainA'
        )
        collector.download_from_unsplash(
            "Hong Kong sunny clear weather urban",
            num_images=args.num_images,
            save_dir='trainB'
        )
        collector.validate_dataset()
        collector.create_info_json()
    elif args.pexels:
        collector.download_from_pexels(
            "Hong Kong rain rainy weather",
            num_images=args.num_images,
            save_dir='trainA'
        )
        collector.download_from_pexels(
            "Hong Kong sunny weather",
            num_images=args.num_images,
            save_dir='trainB'
        )
        collector.validate_dataset()
        collector.create_info_json()
    else:
        collector.print_manual_download_guide()
        print("\n💡 可用命令:")
        print("  python prepare_edhuk_dataset.py --manual    # 显示手动下载指南")
        print("  python prepare_edhuk_dataset.py --bing      # Bing下载")
        print("  python prepare_edhuk_dataset.py --validate  # 验证数据集")


if __name__ == '__main__':
    main()
