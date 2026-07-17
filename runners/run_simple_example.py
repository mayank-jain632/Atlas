from pathlib import Path

from strategies.examples.ma_strategy import MovingAverageStrategy


UID = "ma_QQQ_200"
CAPITAL = 100_000
START_DATE = "2000-01-01"
END_DATE = None


if __name__ == "__main__":
    strategy = MovingAverageStrategy.from_uid(
        uid=UID,
        capital=CAPITAL
    )

    result = strategy.run(
        symbols=[strategy.symbol],
        start=START_DATE,
        end=END_DATE,
    )

    output_dir = Path("results") / UID
    strategy.save_results(output_dir)

    print(f"UID: {UID}")
    print(f"Trades: {len(result['tradebook'])}")
    if not result["equity"].empty:
        print(f"Final equity: {result['equity']['equity'].iloc[-1]:,.2f}")
    print(f"Saved to: {output_dir}")
