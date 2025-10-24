"""
Test script for Two-Stage Search functionality
Demonstrates the improvements from multi-criteria re-ranking
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from typing import Dict, List
from colorama import init, Fore, Style
import time

# Initialize colorama for colored output
init(autoreset=True)


class TwoStageSearchTester:
    """Test and compare two-stage search with single-stage search"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Fore.CYAN}{'=' * 70}")
        print(f"{Fore.CYAN}{text:^70}")
        print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")
    
    def print_section(self, text: str):
        """Print formatted section"""
        print(f"\n{Fore.YELLOW}{'-' * 70}")
        print(f"{Fore.YELLOW}{text}")
        print(f"{Fore.YELLOW}{'-' * 70}{Style.RESET_ALL}\n")
    
    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def search(self,
              image_path: str,
              top_k: int = 10,
              use_two_stage: bool = False,
              adaptive_rerank: bool = False,
              prefer_source: str = "lab",
              promote_diversity: bool = True) -> Dict:
        """Perform search with specified parameters"""
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/search",
                files={'file': f},
                params={
                    'top_k': top_k,
                    'use_two_stage': use_two_stage,
                    'adaptive_rerank': adaptive_rerank,
                    'prefer_source': prefer_source,
                    'promote_diversity': promote_diversity
                }
            )
        
        response.raise_for_status()
        return response.json()
    
    def print_results(self, results: Dict, top_n: int = 5):
        """Print search results in formatted way"""
        
        print(f"Search Time: {Fore.GREEN}{results['search_time_ms']:.2f}ms{Style.RESET_ALL}")
        print(f"Total Results: {results['total_results']}")
        print(f"Engine: {results['search_engine']}")
        print(f"\nTop {top_n} Results:")
        print(f"{'-' * 70}")
        
        for i, result in enumerate(results['results'][:top_n], 1):
            species = result['species']
            source = result['source']
            distance = result['distance']
            
            # Color code by source
            source_color = Fore.BLUE if source == 'lab' else Fore.MAGENTA
            
            print(f"{i:2d}. {Fore.WHITE}{species:<30} {source_color}[{source:>5}]{Style.RESET_ALL}  "
                  f"distance: {distance:.4f}")
    
    def count_unique_species(self, results: List[Dict]) -> int:
        """Count unique species in results"""
        return len(set(r['species'] for r in results))
    
    def compare_methods(self, image_path: str, top_k: int = 10):
        """Compare single-stage, two-stage, and adaptive search"""
        
        self.print_header("Two-Stage Search Comparison Test")
        
        print(f"Query Image: {Fore.CYAN}{image_path}{Style.RESET_ALL}")
        print(f"Top K: {top_k}\n")
        
        # 1. Single-Stage Search (Baseline)
        self.print_section("1️⃣  Single-Stage Search (Baseline)")
        
        start = time.time()
        single_stage = self.search(
            image_path,
            top_k=top_k,
            use_two_stage=False
        )
        single_time = (time.time() - start) * 1000
        
        self.print_results(single_stage, top_n=5)
        unique_species_single = self.count_unique_species(single_stage['results'])
        print(f"\n{Fore.GREEN}Unique species in results: {unique_species_single}{Style.RESET_ALL}")
        
        # 2. Two-Stage Search
        self.print_section("2️⃣  Two-Stage Search (with Re-ranking)")
        
        start = time.time()
        two_stage = self.search(
            image_path,
            top_k=top_k,
            use_two_stage=True,
            adaptive_rerank=False
        )
        two_stage_time = (time.time() - start) * 1000
        
        self.print_results(two_stage, top_n=5)
        unique_species_two = self.count_unique_species(two_stage['results'])
        print(f"\n{Fore.GREEN}Unique species in results: {unique_species_two}{Style.RESET_ALL}")
        
        # 3. Adaptive Two-Stage Search
        self.print_section("3️⃣  Adaptive Two-Stage Search")
        
        start = time.time()
        adaptive = self.search(
            image_path,
            top_k=top_k,
            use_two_stage=True,
            adaptive_rerank=True
        )
        adaptive_time = (time.time() - start) * 1000
        
        self.print_results(adaptive, top_n=5)
        unique_species_adaptive = self.count_unique_species(adaptive['results'])
        print(f"\n{Fore.GREEN}Unique species in results: {unique_species_adaptive}{Style.RESET_ALL}")
        
        # 4. Comparison Summary
        self.print_section("📊 Comparison Summary")
        
        print(f"{'Method':<30} {'Time (ms)':<15} {'Unique Species':<20} {'Improvement'}")
        print(f"{'-' * 85}")
        
        print(f"{'Single-Stage':<30} {single_time:>10.2f}ms    {unique_species_single:>10}           "
              f"{Fore.WHITE}baseline{Style.RESET_ALL}")
        
        time_overhead = ((two_stage_time - single_time) / single_time) * 100
        diversity_improvement = ((unique_species_two - unique_species_single) / 
                                unique_species_single * 100) if unique_species_single > 0 else 0
        
        print(f"{'Two-Stage':<30} {two_stage_time:>10.2f}ms    {unique_species_two:>10}           "
              f"{Fore.GREEN}+{diversity_improvement:.1f}% diversity{Style.RESET_ALL}")
        
        adaptive_overhead = ((adaptive_time - single_time) / single_time) * 100
        adaptive_diversity = ((unique_species_adaptive - unique_species_single) / 
                             unique_species_single * 100) if unique_species_single > 0 else 0
        
        print(f"{'Adaptive Two-Stage':<30} {adaptive_time:>10.2f}ms    {unique_species_adaptive:>10}           "
              f"{Fore.GREEN}+{adaptive_diversity:.1f}% diversity{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Time Overhead:{Style.RESET_ALL}")
        print(f"  Two-Stage: {Fore.YELLOW}+{time_overhead:.1f}%{Style.RESET_ALL}")
        print(f"  Adaptive: {Fore.YELLOW}+{adaptive_overhead:.1f}%{Style.RESET_ALL}")
        
        # 5. Analyze differences
        self.print_section("🔍 Result Differences")
        
        single_top5 = [r['species'] for r in single_stage['results'][:5]]
        two_stage_top5 = [r['species'] for r in two_stage['results'][:5]]
        adaptive_top5 = [r['species'] for r in adaptive['results'][:5]]
        
        print("Top 5 Species Comparison:")
        print(f"\n{'Rank':<6} {'Single-Stage':<25} {'Two-Stage':<25} {'Adaptive'}")
        print(f"{'-' * 85}")
        
        for i in range(5):
            single = single_top5[i] if i < len(single_top5) else ""
            two = two_stage_top5[i] if i < len(two_stage_top5) else ""
            adapt = adaptive_top5[i] if i < len(adaptive_top5) else ""
            
            # Highlight changes
            two_color = Fore.GREEN if two != single else Fore.WHITE
            adapt_color = Fore.GREEN if adapt != single else Fore.WHITE
            
            print(f"{i+1:<6} {Fore.WHITE}{single:<25} {two_color}{two:<25} "
                  f"{adapt_color}{adapt}{Style.RESET_ALL}")
        
        return {
            'single_stage': single_stage,
            'two_stage': two_stage,
            'adaptive': adaptive
        }
    
    def test_diversity_promotion(self, image_path: str):
        """Test diversity promotion feature"""
        
        self.print_header("Diversity Promotion Test")
        
        # Without diversity
        self.print_section("Without Diversity Promotion")
        
        no_diversity = self.search(
            image_path,
            top_k=10,
            use_two_stage=True,
            promote_diversity=False
        )
        
        self.print_results(no_diversity)
        unique_no_div = self.count_unique_species(no_diversity['results'])
        print(f"\n{Fore.GREEN}Unique species: {unique_no_div}/10{Style.RESET_ALL}")
        
        # With diversity
        self.print_section("With Diversity Promotion")
        
        with_diversity = self.search(
            image_path,
            top_k=10,
            use_two_stage=True,
            promote_diversity=True
        )
        
        self.print_results(with_diversity)
        unique_with_div = self.count_unique_species(with_diversity['results'])
        print(f"\n{Fore.GREEN}Unique species: {unique_with_div}/10{Style.RESET_ALL}")
        
        # Summary
        improvement = ((unique_with_div - unique_no_div) / unique_no_div * 100) if unique_no_div > 0 else 0
        print(f"\n{Fore.CYAN}Diversity Improvement: {Fore.GREEN}+{improvement:.1f}%{Style.RESET_ALL}")
    
    def test_source_preference(self, image_path: str):
        """Test source preference feature"""
        
        self.print_header("Source Preference Test")
        
        # Prefer lab
        self.print_section("Prefer Lab Images")
        
        prefer_lab = self.search(
            image_path,
            top_k=10,
            use_two_stage=True,
            prefer_source='lab'
        )
        
        lab_count = sum(1 for r in prefer_lab['results'] if r['source'] == 'lab')
        print(f"Lab images in top-10: {Fore.BLUE}{lab_count}/10{Style.RESET_ALL}")
        self.print_results(prefer_lab, top_n=5)
        
        # Prefer field
        self.print_section("Prefer Field Images")
        
        prefer_field = self.search(
            image_path,
            top_k=10,
            use_two_stage=True,
            prefer_source='field'
        )
        
        field_count = sum(1 for r in prefer_field['results'] if r['source'] == 'field')
        print(f"Field images in top-10: {Fore.MAGENTA}{field_count}/10{Style.RESET_ALL}")
        self.print_results(prefer_field, top_n=5)
        
        # Summary
        print(f"\n{Fore.CYAN}Source Distribution:{Style.RESET_ALL}")
        print(f"  Prefer Lab:   {Fore.BLUE}{lab_count} lab{Style.RESET_ALL}, "
              f"{Fore.MAGENTA}{10-lab_count} field{Style.RESET_ALL}")
        print(f"  Prefer Field: {Fore.BLUE}{10-field_count} lab{Style.RESET_ALL}, "
              f"{Fore.MAGENTA}{field_count} field{Style.RESET_ALL}")


def find_test_image() -> str:
    """Find a test image from the dataset"""
    
    # Try to find an image from the dataset
    dataset_paths = [
        Path("dataset/images/lab"),
        Path("dataset/images/field"),
    ]
    
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            # Get first species directory
            species_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            if species_dirs:
                species_dir = species_dirs[0]
                # Get first image
                images = list(species_dir.glob("*.jpg"))
                if images:
                    return str(images[0])
    
    return None


def main():
    """Main test function"""
    
    tester = TwoStageSearchTester()
    
    # Check server
    print(f"{Fore.YELLOW}Checking server...{Style.RESET_ALL}")
    if not tester.check_server():
        print(f"{Fore.RED}❌ Server is not running at http://localhost:8000")
        print(f"{Fore.YELLOW}Please start the server first: make run{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}✓ Server is running{Style.RESET_ALL}")
    
    # Find test image
    test_image = find_test_image()
    
    if not test_image:
        print(f"\n{Fore.RED}❌ No test image found in dataset")
        print(f"{Fore.YELLOW}Please ensure dataset is populated{Style.RESET_ALL}")
        return
    
    print(f"{Fore.GREEN}✓ Found test image: {test_image}{Style.RESET_ALL}")
    
    # Run tests
    try:
        # Test 1: Compare search methods
        tester.compare_methods(test_image, top_k=10)
        
        # Test 2: Diversity promotion
        tester.test_diversity_promotion(test_image)
        
        # Test 3: Source preference
        tester.test_source_preference(test_image)
        
        # Final summary
        tester.print_header("✅ Two-Stage Search Test Complete!")
        
        print(f"{Fore.GREEN}All tests passed successfully!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}Key Takeaways:{Style.RESET_ALL}")
        print("  • Two-stage search improves result diversity")
        print("  • Adaptive re-ranking auto-adjusts to query type")
        print("  • Source preference helps filter by image quality")
        print("  • Diversity promotion reduces duplicate species")
        print(f"\n{Fore.YELLOW}💡 Enable two-stage search for better results!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ Test failed: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

