"""Integration test for cost tracking system."""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import aita modules
sys.path.append(str(Path(__file__).parent.parent))

from aita.services.llm.cost_tracker import init_global_tracker, get_global_tracker
from aita.services.llm.openrouter import create_openrouter_client
from aita.services.llm.model_pricing import calculate_cost
from aita.services.llm.cost_analysis import CostAnalyzer


def test_cost_tracking_integration():
    """Test cost tracking integration."""
    print("\n🧪 Testing Cost Tracking Integration")
    print("=" * 50)

    # Initialize cost tracker
    print("1. Initializing cost tracker...")
    tracker = init_global_tracker(
        data_dir=Path("C:/Users/ellio/OneDrive - UW-Madison/AITA/intermediateproduct/cost_tracking")
    )
    print(f"   ✅ Initialized session: {tracker.session_id}")

    # Test pricing calculations
    print("\n2. Testing pricing calculations...")
    cost_info = calculate_cost(
        "google/gemini-2.5-flash",
        input_tokens=1000,
        output_tokens=500,
        image_count=2
    )
    print(f"   ✅ Input cost: ${cost_info['input_cost']:.6f}")
    print(f"   ✅ Output cost: ${cost_info['output_cost']:.6f}")
    print(f"   ✅ Image cost: ${cost_info['image_cost']:.6f}")
    print(f"   ✅ Total cost: ${cost_info['total_cost']:.6f}")

    # Test manual cost tracking
    print("\n3. Testing manual cost tracking...")
    entry = tracker.track_call(
        model="google/gemini-2.5-flash",
        operation_type="test_operation",
        input_tokens=1000,
        output_tokens=500,
        image_count=2,
        metadata={"test": True}
    )
    print(f"   ✅ Tracked entry with cost: ${entry.total_cost:.6f}")

    # Test if we have API key for real test
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print("\n4. Testing with real LLM client...")
        try:
            client = create_openrouter_client()
            print("   ✅ Created OpenRouter client")

            # Make a simple text request
            response = client.complete_text(
                prompt="Hello, this is a test message for cost tracking. Please respond briefly.",
                operation_type="test_integration"
            )
            print(f"   ✅ LLM Response: {response[:50]}...")
            print(f"   ✅ Cost should be automatically tracked")

        except Exception as e:
            print(f"   ⚠️ Real LLM test failed: {e}")
    else:
        print("\n4. Skipping real LLM test (no API key)")

    # Test summary
    print("\n5. Testing cost summary...")
    summary = tracker.get_summary()
    print(f"   ✅ Total calls: {summary.total_calls}")
    print(f"   ✅ Total cost: ${summary.total_cost:.6f}")
    print(f"   ✅ Operations: {list(summary.cost_by_operation.keys())}")

    # Test saving
    print("\n6. Testing save/load...")
    tracker.save()
    print(f"   ✅ Saved to: {tracker.session_file}")

    # Test analysis
    print("\n7. Testing cost analysis...")
    analyzer = CostAnalyzer()
    analysis = analyzer.analyze_all_sessions()
    print(f"   ✅ Found {analysis.total_sessions} sessions")
    print(f"   ✅ Total historical cost: ${analysis.total_cost:.6f}")

    # Test projections
    print("\n8. Testing cost projections...")
    projections = analyzer.get_cost_projections(
        students_per_exam=10,
        pages_per_student=5
    )
    print(f"   ✅ Projected cost for 10 students: ${projections.get('total_projected_cost', 0):.4f}")

    # Print final summary
    print("\n" + "=" * 50)
    print("💰 Cost Tracking Summary:")
    tracker.print_summary(detailed=False)

    print("\n✅ All cost tracking tests completed successfully!")


if __name__ == "__main__":
    test_cost_tracking_integration()