"""
Example: Using FINODE Programmatically

This script shows how to integrate FINODE into your own applications.
"""

import asyncio
import json
from finode.api.server import FINODESystem


async def main():
    """Example of direct system usage (without API)"""
    
    print("="*70)
    print("FINODE - Direct System Integration Example")
    print("="*70 + "\n")
    
    # Initialize system
    print("Initializing FINODE system...")
    system = FINODESystem()
    print("✓ System ready\n")
    
    # Example queries
    queries = [
        "What is Apple's recent financial performance?",
        "How much money would $1000 grow to if invested at 7% annually for 5 years?",
        "What is the S&P 500 current status?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print('─'*70)
        
        try:
            # Process query
            result = await system.process_query(query, user_role="analyst")
            
            # Display results
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.0%}")
            print(f"Verified: {result['verified']}")
            print(f"Evidence Count: {len(result['evidence'])}")
            print(f"Execution Time: {result['execution_details']['total_time_ms']:.1f}ms")
            print(f"Audit Hash: {result['audit_hash'][:16]}...")
            
            # Show evidence
            if result['evidence']:
                print("\nEvidence:")
                for idx, evidence in enumerate(result['evidence'], 1):
                    print(f"  {idx}. {evidence}")
            
            # Show LLM stats
            print(f"\nTokens Used:")
            print(f"  Input: {result['llm_stats']['total_input_tokens']}")
            print(f"  Output: {result['llm_stats']['total_output_tokens']}")
        
        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
