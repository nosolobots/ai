package ai.ch2;

import ai.ch2.BaseGameEntity;

public class EnterMineAndDigForNugget implements State {
    private static EnterMineAndDigForNugget instance;

    private EnterMineAndDigForNugget() {}

    public static EnterMineAndDigForNugget getInstance() {
        if (instance == null)
            instance = new EnterMineAndDigForNugget();
        return instance;
    }

    public void enter(BaseGameEntity entity) {
        System.out.println(entity.getName() + ": " + "Walkin' to the gold mine");
        entity.changeLocation(Location.Mine);
    }

    public void exit(BaseGameEntity entity) {
        System.out.println(entity.getName() + ": " + 
                "Ah'm leavin' the gold mine with mah pockets full o' sweet gold");
    }

    public void execute(BaseGameEntity entity) {
        Miner miner = (Miner) entity;
        miner.addToGoldCarried(1); 
        miner.increaseFatigue();
   
        System.out.println(miner.getName() + ": " + "Pickin' up a nugget");

        if(miner.pocketsFull())
            miner.changeState(VisitBankAndDepositGold.getInstance());

        if(miner.thirsty())
            miner.changeState(QuenchThirst.getInstance());
    }
}
