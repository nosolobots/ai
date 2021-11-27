package ai.ch2;

public class Miner extends BaseGameEntity {
    private int goldCarried;
    private int moneyInBank;
    private int thirsty;
    private int fatigue;

    public Miner(int id, String name) { super(id, name); }

    public void addToGoldCarried(int gold) { this.goldCarried += gold; }
    public void increaseFatigue() { ++this.fatigue; }
    
    public void update() {}
}
