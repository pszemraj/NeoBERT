"""
Simple validation script for MuonClip optimizer.
Run: python tests/validate_muonclip.py
"""

import sys
import torch
from neobert.model import NeoBERT, NeoBERTConfig
from neobert.optimizer import MuonClipOptimizer, MuonClipConfig


def validate_basic_functionality():
    """Test basic MuonClip functionality."""
    print("=" * 60)
    print("MuonClip Basic Validation")
    print("=" * 60)

    try:
        # Create tiny model
        config = NeoBERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
            max_position_embeddings=128,
            flash_attention=False,
            hidden_act="gelu",
            rope=False,
        )

        model = NeoBERT(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {num_params / 1e6:.2f}M params")

        # Test with clipping disabled
        print("\n--- Testing Muon-only mode (no clipping) ---")
        muon_config = MuonClipConfig(
            lr=1e-3,
            enable_clipping=False,
        )
        optimizer = MuonClipOptimizer(model, config, muon_config)
        print(
            f"✓ Optimizer created with {len(optimizer.param_groups)} parameter groups"
        )

        # Run one step
        input_ids = torch.randint(0, 1000, (2, 64))
        output = model(input_ids)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"✓ Single step completed, loss={loss.item():.4f}")

        # Test with clipping enabled
        print("\n--- Testing MuonClip with QK-clipping ---")
        model2 = NeoBERT(config)
        muon_config2 = MuonClipConfig(
            lr=1e-3,
            enable_clipping=True,
            clipping_threshold=50.0,
        )
        optimizer2 = MuonClipOptimizer(model2, config, muon_config2)
        print("✓ Optimizer created with hooks enabled")

        # Run one step
        model2.train()  # Important for hooks
        input_ids = torch.randint(0, 1000, (2, 64))
        output = model2(input_ids)
        loss = output.mean()
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()

        # Get metrics
        metrics = optimizer2.get_metrics()
        if "train/max_attention_logit" in metrics:
            print(
                f"✓ Captured max attention logit: {metrics['train/max_attention_logit']:.2f}"
            )
        if "train/attention_entropy" in metrics:
            print(
                f"✓ Captured attention entropy: {metrics['train/attention_entropy']:.2f}"
            )

        print("\n✅ Basic validation passed!")
        return True

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_config():
    """Test configuration validation."""
    print("\n" + "=" * 60)
    print("Configuration Validation")
    print("=" * 60)

    # Test valid config
    try:
        config = MuonClipConfig()
        print(f"✓ Default config created: lr={config.lr}, ns_steps={config.ns_steps}")
    except Exception as e:
        print(f"❌ Default config failed: {e}")
        return False

    # Test invalid configs
    print("\nTesting invalid configurations:")

    # Invalid lr
    try:
        config = MuonClipConfig(lr=0)
        print("❌ Should have rejected lr=0")
        return False
    except AssertionError:
        print("✓ Correctly rejected lr=0")

    # Invalid threshold
    try:
        config = MuonClipConfig(clipping_threshold=0)
        print("❌ Should have rejected clipping_threshold=0")
        return False
    except AssertionError:
        print("✓ Correctly rejected clipping_threshold=0")

    print("\n✅ Configuration validation passed!")
    return True


def validate_newton_schulz():
    """Test Newton-Schulz orthogonalization."""
    print("\n" + "=" * 60)
    print("Newton-Schulz Validation")
    print("=" * 60)

    try:
        # Create optimizer to access Newton-Schulz method
        config = NeoBERTConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
        )
        model = NeoBERT(config)
        muon_config = MuonClipConfig()
        optimizer = MuonClipOptimizer(model, config, muon_config)

        # Test orthogonalization
        G = torch.randn(32, 32)
        X = optimizer._newton_schulz_update(G, steps=5)

        print(f"✓ Input matrix norm: {G.norm():.2f}")
        print(f"✓ Output matrix norm: {X.norm():.2f}")

        # Check that output is reasonable
        if X.norm() > 0 and X.norm() < G.norm() * 100:
            print("✓ Newton-Schulz produces reasonable output")
        else:
            print("❌ Newton-Schulz output seems wrong")
            return False

        print("\n✅ Newton-Schulz validation passed!")
        return True

    except Exception as e:
        print(f"\n❌ Newton-Schulz validation failed: {e}")
        return False


def validate_hook_system():
    """Test attention hook system."""
    print("\n" + "=" * 60)
    print("Hook System Validation")
    print("=" * 60)

    try:
        from neobert.optimizer.muon_clip import NeoBERTAttentionHooks

        config = NeoBERTConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            vocab_size=1000,
            max_position_embeddings=128,
            flash_attention=False,
            rope=False,
        )
        model = NeoBERT(config)

        # Create and register hooks
        hook_system = NeoBERTAttentionHooks(config, offload_to_cpu=False)
        num_hooks = hook_system.register_hooks(model)
        print(f"✓ Registered {num_hooks} hooks")

        # Run forward pass
        model.train()
        input_ids = torch.randint(0, 1000, (2, 64))
        _ = model(input_ids)

        # Check captured data
        if len(hook_system.layer_stats) == 2:
            print(f"✓ Captured stats from {len(hook_system.layer_stats)} layers")

            # Check layer 0 stats
            stats = hook_system.get_layer_stats(0)
            if stats and "max_logits_per_head" in stats:
                print(f"✓ Layer 0 max logit: {stats['max_logit_overall']:.2f}")
                print(f"✓ Layer 0 per-head shape: {stats['max_logits_per_head'].shape}")

        # Clean up
        hook_system.clear()
        hook_system.remove_hooks()
        print("✓ Hooks cleaned up successfully")

        print("\n✅ Hook system validation passed!")
        return True

    except Exception as e:
        print(f"\n❌ Hook system validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    all_passed = True

    # Run all validations
    all_passed &= validate_config()
    all_passed &= validate_newton_schulz()
    all_passed &= validate_hook_system()
    all_passed &= validate_basic_functionality()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All validations passed successfully!")
        print("MuonClip optimizer is ready for use!")
    else:
        print("❌ Some validations failed. Please check the errors above.")
        sys.exit(1)
    print("=" * 60)
